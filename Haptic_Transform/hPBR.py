import struct
from io import BytesIO
import numpy as np
from PIL import Image
import hashlib
import zlib


magic_num = bytes([0x68,0x50,0x42,0x52]) # "hPBR".encode().hex()

version = 1

class hPBR:
    def __init__(self, file_path, material_properties_dict, pbr):
        self.path = file_path
        self.material_prop = material_properties_dict
        self.pbr = pbr

    def transform(self):
        with open(self.path, 'wb') as f:
            f.write(magic_num)
            f.write(bytes([version]))
            for prop in self.material_prop:
                payload = self.numpy_array_payload_generator(self.material_prop[prop])
                self.write_chunk(f,b"NARR",prop,payload)
            for tile_map in self.pbr.tile_maps:
                payload = self.image_payload_generator(self.pbr.tile_maps[tile_map])
                self.write_chunk(f,b"IMAG",tile_map,payload)

        self.append_integrity_hash()
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

    # Expected payload, no bigger than 4GB.
    def image_payload_generator(self, img):
        buf = BytesIO()
        img.save(buf, format='PNG')
        raw_data = buf.getvalue()

        header = struct.pack('B', 1) # 1-byte format code = 1 for PNG
        return header + raw_data

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


    def append_integrity_hash(self):
        """
        After writing all chunks, compute SHA-256 of the entire file
        (excluding the final 32 bytes that hold the hash itself),
        then append the 32-byte hash at the end.
        """
        
        hashed = hashlib.sha256()
        with open(self.path, 'rb') as f:
            chunk = f.read(4096) #Remember we are around 4B chunks.
            while chunk:
                hashed.update(chunk)
                chunk = f.read(4096)

        # Appending the 32B sha256 hash to the end of the file.
        with open(self.path, 'ab') as f:
            f.write(hashed.digest())