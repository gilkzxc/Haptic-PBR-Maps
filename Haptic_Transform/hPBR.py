import struct
from io import BytesIO
import numpy as np
from PIL import Image
import hashlib
import zlib
import os

from PBR_Modules.PBR import PBR


magic_num = bytes([0x68,0x50,0x42,0x52]) # "hPBR".encode().hex()

version = 1

class hPBR:
    def __init__(self, file_path = "", material_properties_dict = {}, pbr = PBR()):
        self.path = file_path
        self.material_prop = material_properties_dict
        self.pbr = pbr

    def transform(self):
        if self.path == "" and self.material_prop:
            return
        with open(self.path, 'wb') as f:
            f.write(magic_num)
            f.write(bytes([version]))
            for prop in self.material_prop:
                if isinstance(self.material_prop[prop], np.ndarray):
                    payload = self.numpy_array_payload_generator(self.material_prop[prop])
                    self.write_chunk(f,b"NARR",prop,payload)
            for tile_map in self.pbr.tile_maps:
                if isinstance(self.pbr.tile_maps[tile_map], Image.Image):
                    payload = self.image_payload_generator(self.pbr.tile_maps[tile_map])
                    self.write_chunk(f,b"IMAG",tile_map,payload)

        self.append_integrity_hash()

    def fromFile(self, file_path):
        filesize = os.path.getsize(file_path)

        if not verify_file(file_path,filesize):
            raise ValueError(f"File: {file_path} isn't hPBR format.")

        material_prop = {}
        tile_maps = {}

        with open(file_path, 'rb') as f:
            magicNum = f.read(4)
            ver = f.read(1)
            if magicNum != magic_num or ver != bytes([version]):
                raise ValueError(f"File: {file_path} isn't hPBR format.")
            try:
                for tag, name, payload in self.read_chunks(f,filesize):
                    if tag == b"NARR":
                        try:
                            material_prop[name] = self.numpy_payload_parser(payload)
                        except ValueError as e:
                            raise ValueError(f"{name} property map is invalid dtype.")
                    elif tag == b"IMAG":
                        try:
                            tile_maps[name] = self.image_payload_parser(payload)
                        except ValueError as e:
                            raise ValueError(f"{name} tile map isn't a PNG image.")
            except ValueError as e:
                raise ValueError(f"File: {file_path} isn't hPBR format: {e}")

        self.path = file_path
        self.material_prop = material_prop
        self.pbr = PBR(tile_maps)

    def write_chunk(self, f, tag, name, payload):

        # Tag: e.g. b"NARR" or b"IMAG" (4 bytes)
        f.write(tag)

        length_pos = f.tell()
        f.write(struct.pack('<I', 0))  # placeholder for length as a 4-byte integer

        # Attaching name
        name_bytes = name.encode('utf-8')
        f.write(struct.pack('B', len(name_bytes)))
        f.write(name_bytes)

        # Write payload
        start_payload_pos = f.tell()
        f.write(payload)
        end_payload_pos = f.tell()

        # Compute CRC32 of the payload
        crc_val = zlib.crc32(payload) & 0xffffffff  # ensure unsigned 32-bit
        f.write(struct.pack('<I', crc_val))

        end_pos = f.tell()

        # update the chunk length = payload_size + CRC_size
        chunk_length = (end_payload_pos - start_payload_pos) + 4
        f.seek(length_pos)
        f.write(struct.pack('<I', chunk_length))
        f.seek(end_pos)

    def read_chunks(self, f, file_size):
        """
        Reads chunk after chunk until the position of hash.
        Verifies the CRC of each chunk's payload.
        Yields (tag, name, payload) tuples if CRC is valid.
        Raises ValueError if a CRC mismatch occurs.
        """
        while f.tell() < file_size - 32:
            tag = f.read(4)
            if not tag:
                # Reached end of chunk stream
                break

            chunk_len_bytes = f.read(4)
            if not chunk_len_bytes or len(chunk_len_bytes) < 4:
                # Incomplete data (corrupt or unexpected EOF)
                raise ValueError("File ended unexpectedly while reading chunk length.")
            chunk_length = struct.unpack('<I', chunk_len_bytes)[0]

            name_len_data = f.read(1)
            if not name_len_data:
                raise ValueError("File ended unexpectedly while reading name length.")
            name_len = struct.unpack('B', name_len_data)[0]

            name_data = f.read(name_len)
            if len(name_data) < name_len:
                raise ValueError("File ended unexpectedly while reading name.")

            name = name_data.decode('utf-8')
            payload_size = chunk_length - 4
            if payload_size < 0:
                raise ValueError(f"Invalid chunk length: {chunk_length}")

            payload = f.read(payload_size)
            if len(payload) < payload_size:
                raise ValueError(f"Incomplete data for payload of '{name}' chunk.")

            crc_data = f.read(4)
            if len(crc_data) < 4:
                raise ValueError("File ended unexpectedly while reading CRC.")
            stored_crc = struct.unpack('<I', crc_data)[0]
            computed_crc = zlib.crc32(payload) & 0xffffffff
            if computed_crc != stored_crc:
                raise ValueError(f"CRC mismatch for chunk '{name}'. "
                                 f"Expected 0x{stored_crc:08X}, got 0x{computed_crc:08X}.")
            yield (tag, name, payload)

    # Expected payload, no bigger than 4GB.
    def image_payload_generator(self, img):
        buf = BytesIO()
        img.save(buf, format='PNG')
        raw_data = buf.getvalue()

        header = struct.pack('B', 1) # 1-byte format code = 1 for PNG
        return header + raw_data

    def image_payload_parser(self, payload):
        format_code = payload[0]
        raw_data = payload[1:]
        if format_code == 1:
            # PNG
            return Image.open(BytesIO(raw_data))
        raise ValueError(f"Unsupported image format code {format_code}")

    def numpy_array_payload_generator(self, narr):

        rows, cols = narr.shape

        dtype_map = {
            np.float32: 1,
            np.float64: 2,
            np.int32:  3,
            np.int64:  4
        }

        code = dtype_map.get(narr.dtype.type, 0)  # 0 = unknown

        # convert to known dtype if needed
        if code == 0:
            narr = narr.astype(np.float64)
            code = 2

        # build payload
        header = struct.pack('<IIb', rows, cols, code)
        # raw data
        data = narr.tobytes(order='C')
        return header + data

    def numpy_payload_parser(self,payload):
    
        # read first 4+4+1 = 9 bytes for (rows, cols, dtype_code)
        rows, cols, code = struct.unpack('<IIb', payload[:9])
        
        dtype_map = {
            1: np.float32,
            2: np.float64,
            3:  np.int32,
            4:  np.int64
        }
        dtype = dtype_map.get(code, None)
        if dtype is None:
            raise ValueError(f"Unknown dtype code {code}")

        data = payload[9:]
        arr = np.frombuffer(data, dtype=dtype).reshape((rows, cols), order='C')
        return arr

    def append_integrity_hash(self):
        """
        After writing all chunks, compute SHA-256 of the entire file
        (excluding the final 32 bytes that hold the hash itself),
        then append the 32-byte hash at the end.
        """
        
        hashed = hashlib.sha256()
        with open(self.path, 'rb') as f:
            chunk = f.read(4096) # Reading 4KB chuncks as a usual page size.
            while chunk:
                hashed.update(chunk)
                chunk = f.read(4096)

        # Appending the 32B sha256 hash to the end of the file.
        with open(self.path, 'ab') as f:
            f.write(hashed.digest())
    
def verify_file(filepath, filesize):

    
    if filesize < 32:
        return False

       
    with open(filepath, 'rb') as f:
        f.seek(filesize - 32)
        stored_hash = f.read(32)

    
    sha256_hash = hashlib.sha256()
    with open(filepath, 'rb') as f:
        bytes_to_read = filesize - 32
        for chunk in iter(lambda: f.read(4096 if bytes_to_read > 4096 else bytes_to_read), b''):
            if not chunk:
                break
            sha256_hash.update(chunk)
            bytes_to_read -= len(chunk)

    computed_hash = sha256_hash.digest()

    return computed_hash == stored_hash
    
        
