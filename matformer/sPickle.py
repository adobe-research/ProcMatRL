# BSD license: https://opensource.org/license/BSD-3-Clause

'''Streaming pickle implementation for efficiently serializing and
de-serializing an iterable (e.g., list)

Created on 2010-06-19 by Philip Guo
Mostly rewritten 2015-01-16 by Dustin King for Python 3.4
Mostly rewritten 2023-10-02 by Beichen Li (chunked I/O)

Not backwards compatible.
'''

from pickle import dumps, loads
from typing import BinaryIO, Iterable, Iterator, Any

# Global constants
BS, NL = ord('\\'), ord('\n')


def writeByteArray(byteArray: bytes, binaryFile: BinaryIO):
    '''Write a bytearray or bytes object to a file,
    escaping so that multiple bytearray's can be writen.
    filecontents  -> array contents
    \\ -> \
    \\n -> \n
    \n -> end of byte array
    '''
    f = binaryFile
    p = 0   # last unprocessed position

    # escape reserved characters
    for i, b in enumerate(byteArray):
        if b == BS or b == NL:
            f.write(byteArray[p:i])
            f.write(bytes([BS, b]))
            p = i + 1

    # write remaining data
    if p < len(byteArray):
        f.write(byteArray[p:])
    f.write(b'\n')

    # original code for reference
    # for byte in byteArray:
    #     if byte == b'\\'[0]:
    #         binaryFile.write(b'\\\\')
    #     elif byte == b'\n'[0]:
    #         binaryFile.write(b'\\\n')
    #     else:
    #         binaryFile.write(bytes([byte]))
    # binaryFile.write(b'\n')


def writeByteArrayStream(byteArrays: Iterable[bytes], binaryFile: BinaryIO):
    for barray in byteArrays:
        writeByteArray(barray, binaryFile)

def readByteArrayStream(binaryFile: BinaryIO, chunksize: int) -> Iterator[bytes]:
    f = binaryFile
    buf = bytearray()
    escape = False  # escape switch

    while True:
        # read the next chunk
        chunk = f.read(chunksize)
        if not chunk:
            break

        # process chunk
        p = 0  # last unprocessed position in the chunk

        for i, b in enumerate(chunk):
            if escape:
                if b == BS or b == NL:
                    buf.append(b)
                    p, escape = i + 1, False
                else:
                    raise RuntimeError(f"Unexpected byte: '{str(b)}' (ord = 0x{b:x})")
            elif b == BS:
                buf.extend(chunk[p:i])
                p, escape = i + 1, True
            elif b == NL:
                buf.extend(chunk[p:i])
                yield bytes(buf)
                buf.clear()
                p = i + 1

        if p < len(chunk):
            buf.extend(chunk[p:])

    if escape:
        raise RuntimeError("Unexpected end with the escape byte")

    # original code for reference
    # byte = f.read(1)
    # while byte != b'':
    #     if byte == b'\\':
    #         byte = f.read(1)
    #         if byte == b'\\':
    #             buf.append(b'\\'[0])
    #         elif byte == b'\n':
    #             buf.append(b'\n'[0])
    #         else:
    #             raise Exception('unexpected byte: ' + str(byte))
    #     elif byte == b'\n':
    #         yield bytes(buf)
    #         buf = bytearray()
    #     else:
    #         buf.append(byte[0])
    #     byte = f.read(1)

def pickleIterable(iterable: Iterable[Any]) -> Iterator[bytes]:
    for item in iterable:
        yield dumps(item)


def s_dump(iterable_to_pickle: Iterable[Any], file_obj: BinaryIO):
    '''dump contents of an iterable iterable_to_pickle to file_obj, a file
    opened in write mode'''
    writeByteArrayStream(pickleIterable(iterable_to_pickle), file_obj)

def s_dump_elt(elt: Any, file_obj: BinaryIO):
    writeByteArray(dumps(elt), file_obj)

def s_load(file_obj: BinaryIO, chunksize: int = 4096) -> Iterator[Any]:
    '''load contents from file_obj, returning a generator that yields one
    element at a time'''
    if not isinstance(chunksize, int) or chunksize <= 0:
        raise ValueError(f'Invalid chunks size: {chunksize}')
    for barray in readByteArrayStream(file_obj, chunksize):
        yield loads(barray)
