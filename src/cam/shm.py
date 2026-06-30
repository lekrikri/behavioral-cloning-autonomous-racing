"""
shm.py — Ring buffer mémoire partagée, 1 writer / N readers, sémantique "dernière frame".

Transport LOCAL du hub caméra. Le producteur (hub) écrit chaque frame UNE fois dans un
ring en /dev/shm (tmpfs = RAM) ; les consommateurs locaux la lisent en ZÉRO-COPIE (vue
numpy directe sur la mmap). Pas de pile réseau, pas de copie par client — décisif sur le
Jetson Nano (memory-bandwidth-bound) avec plusieurs consommateurs simultanés.

Modèle :
  - 1 seul writer, N readers, pas de file : on ne sert que la dernière frame.
  - Le writer tourne sur `slot_count` slots (slot = seq % slot_count). Il écrit le payload
    PUIS l'entête du slot (avec seq), PUIS publie `latest_seq` EN DERNIER.
  - Seqlock : un reader lit `latest_seq`, vise slot = seq % count, lit l'entête du slot et
    revérifie que `slot.seq == seq` APRÈS avoir pris sa vue. Si ça diffère, le writer a
    relapé ce slot pendant la lecture → on rejette (lecture déchirée) et on réessaie.
  - Avec slot_count >= 3 et des readers temps réel (lisent chaque frame), la marge avant
    qu'un slot soit réécrit est slot_count * intervalle_frame (~4*33ms = 132ms à 30 fps).

Compatible Python 3.6 (mmap + struct ; PAS de multiprocessing.shared_memory, absent en 3.6).
"""

import mmap
import os
import struct

import numpy as np

_MAGIC = b"RCH2"

# Entête global du segment (offset 0). Padé à 64 octets pour isoler latest_seq.
#   magic(4) version(u16) slot_count(u16) slot_bytes(u32) latest_seq(u64)
_GLOBAL = struct.Struct("<4sHHIQ")
_GLOBAL_SIZE = 64

# Entête par slot. Padé à 40 octets.
#   seq(u64) stream_id(u8) dtype(u8) channels(u8) _pad(u8) h(u16) w(u16) ts_ns(u64) nbytes(u32)
_SLOT_HDR = struct.Struct("<QBBBBHHQI")
_SLOT_HDR_SIZE = 40

_DTYPE_CODE = {np.dtype(np.uint8): 0, np.dtype(np.uint16): 1, np.dtype(np.float32): 2}
_CODE_DTYPE = {0: np.uint8, 1: np.uint16, 2: np.float32}

_SHM_DIR = "/dev/shm"


def _path(name):
    return os.path.join(_SHM_DIR, name)


def _slot_offset(slot_idx, slot_bytes):
    return _GLOBAL_SIZE + slot_idx * (_SLOT_HDR_SIZE + slot_bytes)


def _total_size(slot_count, slot_bytes):
    return _GLOBAL_SIZE + slot_count * (_SLOT_HDR_SIZE + slot_bytes)


class ShmRingWriter:
    """Producteur : crée/écrase le segment et publie les frames. Un seul writer par segment."""

    def __init__(self, name, slot_bytes, slot_count=4):
        self.name = name
        self.slot_bytes = int(slot_bytes)
        self.slot_count = int(slot_count)
        size = _total_size(self.slot_count, self.slot_bytes)
        path = _path(name)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, size)
            self.mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        self._seq = 0
        self.mm.seek(0)
        _GLOBAL.pack_into(self.mm, 0, _MAGIC, 1, self.slot_count, self.slot_bytes, 0)

    def write(self, frame, stream_id=0, ts_ns=0):
        """Publie `frame` (ndarray C-contigu) dans le prochain slot. Retourne le seq publié."""
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        nbytes = frame.nbytes
        if nbytes > self.slot_bytes:
            raise ValueError("frame {}o > slot_bytes {}o".format(nbytes, self.slot_bytes))
        h = frame.shape[0]
        w = frame.shape[1]
        ch = frame.shape[2] if frame.ndim == 3 else 1
        dcode = _DTYPE_CODE[frame.dtype]

        seq = self._seq + 1
        slot = seq % self.slot_count
        base = _slot_offset(slot, self.slot_bytes)
        payload_off = base + _SLOT_HDR_SIZE

        # 1) payload, 2) entête du slot avec le seq, 3) publication de latest_seq EN DERNIER.
        # Copie DIRECTE ndarray → mmap (1 seul memcpy, zéro alloc) ; frame est déjà C-contigu.
        np.frombuffer(self.mm, dtype=np.uint8, count=nbytes, offset=payload_off)[:] = \
            frame.reshape(-1).view(np.uint8)
        _SLOT_HDR.pack_into(self.mm, base, seq, stream_id, dcode, ch, 0, h, w, ts_ns, nbytes)
        _GLOBAL.pack_into(self.mm, 0, _MAGIC, 1, self.slot_count, self.slot_bytes, seq)
        self._seq = seq
        return seq

    def close(self):
        try:
            self.mm.close()
        except Exception:
            pass

    def unlink(self):
        self.close()
        try:
            os.unlink(_path(self.name))
        except OSError:
            pass


class ShmRingReader:
    """Consommateur : attache un segment existant et lit la dernière frame en zéro-copie."""

    def __init__(self, name):
        self.name = name
        path = _path(name)
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            self.mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
        finally:
            os.close(fd)
        magic, ver, self.slot_count, self.slot_bytes, _ = _GLOBAL.unpack_from(self.mm, 0)
        if magic != _MAGIC:
            raise ValueError("segment SHM invalide: {}".format(self.name))
        self._last_seen = 0

    def latest_seq(self):
        return _GLOBAL.unpack_from(self.mm, 0)[4]

    def read(self, copy=False, _retries=4):
        """Retourne (meta, ndarray) de la dernière frame, ou None si rien de neuf/illisible.

        meta = dict(seq, stream_id, h, w, channels, dtype, ts_ns).
        Par défaut la frame est une VUE zéro-copie sur la mmap (lecture seule) : le writer
        peut la réécrire après slot_count frames. Mets copy=True pour la garder longtemps.
        """
        for _ in range(_retries):
            seq = self.latest_seq()
            if seq == 0:
                return None
            slot = seq % self.slot_count
            base = _slot_offset(slot, self.slot_bytes)
            sseq, stream_id, dcode, ch, _pad, h, w, ts_ns, nbytes = _SLOT_HDR.unpack_from(self.mm, base)
            if sseq != seq:
                continue  # writer en cours d'écriture de ce slot → réessai
            dtype = _CODE_DTYPE[dcode]
            payload_off = base + _SLOT_HDR_SIZE
            arr = np.frombuffer(self.mm, dtype=dtype, count=nbytes // np.dtype(dtype).itemsize,
                                offset=payload_off)
            arr = arr.reshape((h, w, ch)) if ch > 1 else arr.reshape((h, w))
            if copy:
                arr = arr.copy()
            # Seqlock : revérifie APRÈS la copie que le slot n'a pas été relapé entre-temps.
            # (En copy=False, valide que la vue était stable jusqu'ici ; le caller doit
            #  consommer promptement, le writer relape après slot_count frames.)
            if _SLOT_HDR.unpack_from(self.mm, base)[0] != seq:
                continue
            self._last_seen = seq
            meta = {"seq": seq, "stream_id": stream_id, "h": h, "w": w,
                    "channels": ch, "dtype": dtype, "ts_ns": ts_ns}
            return meta, arr
        return None

    def close(self):
        try:
            self.mm.close()
        except Exception:
            pass
