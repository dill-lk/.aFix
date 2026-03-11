/**
 * Tests for afix-decoder.js
 *
 * Builds minimal valid .aFix binary buffers in JavaScript and exercises the
 * AfixDecoder / AfixFile API.
 */

import { AfixDecoder } from './afix-decoder.js';

// ── Helpers ────────────────────────────────────────────────────────────────────

const MAGIC       = Buffer.from([0x41, 0x46, 0x49, 0x58, 0x4B]); // AFIXK
const ATOM_MAP    = Buffer.alloc(144, 0);
const PAYLOAD_OFF = 0xB1; // 177

/**
 * CRC-32 (IEEE 802.3) — mirrors the implementation in afix-decoder.js.
 * @param {Uint8Array} data
 * @returns {number}
 */
function crc32(data) {
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < data.length; i++) {
    crc ^= data[i];
    for (let j = 0; j < 8; j++) {
      crc = (crc >>> 1) ^ ((crc & 1) ? 0xEDB88320 : 0);
    }
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

/**
 * Build a minimal valid .aFix file buffer.
 *
 * @param {{width?: number, height?: number, chunks?: Array<{id: string, data: Buffer}>}} opts
 * @returns {ArrayBuffer}
 */
function buildAfixBuffer({ width = 1920, height = 1080, chunks = [] } = {}) {
  // ── Header ────────────────────────────────────────────────────────────────
  const header = Buffer.allocUnsafe(33);
  MAGIC.copy(header, 0);               // 5 B magic
  header.writeUInt8(1, 5);             // major
  header.writeUInt8(0, 6);             // minor
  header.writeUInt8(4, 7);             // patch
  header.writeUInt8(0, 8);             // flag
  header.writeDoubleLE(width,  9);     // width  (f64 LE)
  header.writeDoubleLE(height, 17);    // height (f64 LE)
  header.fill(0, 25, 33);              // 8 B reserved

  // ── Chunk payloads ────────────────────────────────────────────────────────
  const chunkBuffers = chunks.map(({ id, data }) => {
    const idBuf  = Buffer.from(id.padEnd(4, '_').slice(0, 4), 'ascii');
    const lenBuf = Buffer.allocUnsafe(4);
    lenBuf.writeUInt32LE(data.length);
    const flagsBuf = Buffer.alloc(4, 0); // flags (2B) + reserved (2B)
    const crcBuf   = Buffer.allocUnsafe(4);
    crcBuf.writeUInt32LE(crc32(data));
    return Buffer.concat([idBuf, lenBuf, flagsBuf, data, crcBuf]);
  });

  const payloadBuf = Buffer.concat(chunkBuffers);

  // ── Pad to PAYLOAD_OFF ───────────────────────────────────────────────────
  const prePad = Buffer.alloc(PAYLOAD_OFF - header.length - ATOM_MAP.length, 0);
  const full   = Buffer.concat([header, ATOM_MAP, prePad, payloadBuf]);

  return full.buffer.slice(full.byteOffset, full.byteOffset + full.byteLength);
}

// ── Tests ──────────────────────────────────────────────────────────────────────

describe('AfixDecoder', () => {
  test('accepts a valid file and returns version info', () => {
    const buf     = buildAfixBuffer();
    const decoder = new AfixDecoder(buf);
    const file    = decoder.decode();
    expect(file.version.major).toBe(1);
    expect(file.version.minor).toBe(0);
    expect(file.version.patch).toBe(4);
  });

  test('rejects a buffer with wrong magic bytes', () => {
    const buf   = buildAfixBuffer();
    const bytes = new Uint8Array(buf);
    bytes[0] = 0x00; // corrupt magic
    expect(() => new AfixDecoder(buf).decode()).toThrow(/magic/i);
  });

  test('parses width and height from DESC block', () => {
    const buf  = buildAfixBuffer({ width: 800, height: 600 });
    const file = new AfixDecoder(buf).decode();
    expect(file.dimensions.width).toBeCloseTo(800);
    expect(file.dimensions.height).toBeCloseTo(600);
  });

  test('parses META chunk and exposes it as JSON', () => {
    const metaJson = JSON.stringify({ version: '1.0', creator: 'test', profile: 'web-lossy' });
    const buf      = buildAfixBuffer({
      chunks: [{ id: 'META', data: Buffer.from(metaJson, 'utf8') }],
    });
    const file = new AfixDecoder(buf).decode();
    expect(file.meta).not.toBeNull();
    expect(file.meta.creator).toBe('test');
    expect(file.meta.profile).toBe('web-lossy');
  });

  test('parses OBJM chunk and exposes object manifest', () => {
    const manifest = JSON.stringify({
      version: '1.0',
      objects: [
        { id: 'sky',    label: 'sky',        category: 'background' },
        { id: 'face_0', label: 'human_face', category: 'subject' },
      ],
    });
    const buf  = buildAfixBuffer({
      chunks: [{ id: 'OBJM', data: Buffer.from(manifest, 'utf8') }],
    });
    const file = new AfixDecoder(buf).decode();
    expect(file.objectManifest).not.toBeNull();
    expect(file.objectManifest.objects).toHaveLength(2);
    expect(file.objectManifest.objects[0].id).toBe('sky');
  });

  test('detects CRC mismatch and throws', () => {
    const data     = Buffer.from('hello');
    const buf      = buildAfixBuffer({ chunks: [{ id: 'META', data }] });
    const bytes    = new Uint8Array(buf);
    // Corrupt the chunk data byte (offset 0xB1 + 12 = first data byte).
    bytes[0xB1 + 12] ^= 0xFF;
    expect(() => new AfixDecoder(buf).decode()).toThrow(/CRC/i);
  });

  test('getChunk returns null for absent chunk ID', () => {
    const buf  = buildAfixBuffer();
    const file = new AfixDecoder(buf).decode();
    expect(file.getChunk('SIGB')).toBeNull();
  });

  test('handles file with multiple chunks', () => {
    const buf = buildAfixBuffer({
      chunks: [
        { id: 'META', data: Buffer.from('{}') },
        { id: 'VEC_', data: Buffer.from([0, 0, 0, 0]) }, // count=0
        { id: 'OBJM', data: Buffer.from('{"version":"1.0","objects":[]}') },
      ],
    });
    const file = new AfixDecoder(buf).decode();
    expect(file.chunks).toHaveLength(3);
    expect(file.getChunk('META')).not.toBeNull();
    expect(file.getChunk('VEC_')).not.toBeNull();
    expect(file.getChunk('OBJM')).not.toBeNull();
  });

  // ── PREV chunk (JPEG backward-compat preview) ─────────────────────────────

  test('PREV chunk is exposed via previewJpeg getter', () => {
    // Minimal JPEG magic bytes: FF D8 FF E0 ...
    const fakeJpeg = Buffer.from([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x01]);
    const buf  = buildAfixBuffer({
      chunks: [
        { id: 'META', data: Buffer.from('{}') },
        { id: 'PREV', data: fakeJpeg },
      ],
    });
    const file = new AfixDecoder(buf).decode();
    expect(file.previewJpeg).not.toBeNull();
    expect(file.previewJpeg[0]).toBe(0xFF);
    expect(file.previewJpeg[1]).toBe(0xD8);
    expect(file.previewJpeg[2]).toBe(0xFF);
  });

  test('previewJpeg returns null when PREV chunk absent', () => {
    const buf  = buildAfixBuffer({ chunks: [{ id: 'META', data: Buffer.from('{}') }] });
    const file = new AfixDecoder(buf).decode();
    expect(file.previewJpeg).toBeNull();
  });

  test('PREV chunk can coexist with other chunks', () => {
    const fakeJpeg = Buffer.from([0xFF, 0xD8, 0xFF]);
    const metaJson = JSON.stringify({ version: '1.0', s2_codec: 'dct' });
    const buf  = buildAfixBuffer({
      chunks: [
        { id: 'META', data: Buffer.from(metaJson) },
        { id: 'PREV', data: fakeJpeg },
        { id: 'VEC_', data: Buffer.from([0, 0, 0, 0]) },
      ],
    });
    const file = new AfixDecoder(buf).decode();
    expect(file.chunks).toHaveLength(3);
    expect(file.previewJpeg).not.toBeNull();
    expect(file.meta.s2_codec).toBe('dct');
    expect(file.getChunk('VEC_')).not.toBeNull();
  });
});
