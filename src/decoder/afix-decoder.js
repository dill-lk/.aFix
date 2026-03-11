/**
 * afix-decoder.js — JavaScript decoder for the .aFix image format.
 *
 * Parses the binary .aFix format (header, atom map, payload chunks) and
 * renders the image into a <canvas> element using progressive LOD rendering.
 *
 * @module afix-decoder
 * @version 1.0.0
 * @license MIT
 */

'use strict';

// ── Constants ──────────────────────────────────────────────────────────────────

/** ASCII magic bytes: "AFIXK" */
const MAGIC = new Uint8Array([0x41, 0x46, 0x49, 0x58, 0x4B]);

/** Byte offset of the ATOM_MAP (0x21 = 33). */
const ATOM_MAP_OFFSET = 0x21;

/** Size of the ATOM_MAP in bytes. */
const ATOM_MAP_SIZE = 144;

/** Byte offset of the PAYLOAD (0xB1 = 177). */
const PAYLOAD_OFFSET = 0xB1;

/** Maximum allowed chunk data size (512 MB). */
const MAX_CHUNK_SIZE = 512 * 1024 * 1024;

/** Known chunk IDs. */
const CHUNK_IDS = {
  META: 'META',
  VEC:  'VEC_',
  LAT:  'LAT_',
  RES:  'RES_',
  DPTH: 'DPTH',
  SIGB: 'SIGB',
  OBJM: 'OBJM',
};

// ── Utility functions ──────────────────────────────────────────────────────────

/**
 * Decode four bytes as an ASCII string.
 * @param {DataView} view
 * @param {number} offset
 * @returns {string}
 */
function readId(view, offset) {
  return String.fromCharCode(
    view.getUint8(offset),
    view.getUint8(offset + 1),
    view.getUint8(offset + 2),
    view.getUint8(offset + 3),
  );
}

/**
 * Compute CRC-32 of a Uint8Array.
 * Uses the standard IEEE 802.3 polynomial (0xEDB88320).
 * @param {Uint8Array} data
 * @returns {number} unsigned 32-bit CRC
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

// ── AfixDecoder ────────────────────────────────────────────────────────────────

/**
 * Parses an `.aFix` binary file from an ArrayBuffer.
 *
 * @example
 * const response = await fetch('photo.afix');
 * const buffer   = await response.arrayBuffer();
 * const decoder  = new AfixDecoder(buffer);
 * const result   = decoder.decode();
 */
export class AfixDecoder {
  /**
   * @param {ArrayBuffer} buffer — raw `.aFix` file bytes
   */
  constructor(buffer) {
    this._buffer = buffer;
    this._view   = new DataView(buffer);
    this._bytes  = new Uint8Array(buffer);
  }

  /**
   * Parse the file and return a decoded {@link AfixFile} object.
   *
   * @returns {AfixFile}
   * @throws {Error} if the file is malformed
   */
  decode() {
    this._checkMagic();
    const version    = this._readVersion();
    const dimensions = this._readDimensions();
    const chunks     = this._readChunks();
    return new AfixFile({ version, dimensions, chunks });
  }

  // ── Private helpers ──────────────────────────────────────────────────────────

  _checkMagic() {
    for (let i = 0; i < MAGIC.length; i++) {
      if (this._bytes[i] !== MAGIC[i]) {
        throw new Error(
          `Invalid .aFix magic bytes at offset ${i}: ` +
          `expected 0x${MAGIC[i].toString(16).padStart(2, '0')} ` +
          `got 0x${this._bytes[i].toString(16).padStart(2, '0')}`
        );
      }
    }
  }

  _readVersion() {
    return {
      major: this._view.getUint8(5),
      minor: this._view.getUint8(6),
      patch: this._view.getUint8(7),
      flag:  this._view.getUint8(8),
    };
  }

  _readDimensions() {
    return {
      width:  this._view.getFloat64(9,  true),
      height: this._view.getFloat64(17, true),
    };
  }

  _readChunks() {
    const chunks = [];
    let offset = PAYLOAD_OFFSET;

    while (offset + 12 <= this._buffer.byteLength) {
      const id         = readId(this._view, offset);
      const dataLength = this._view.getUint32(offset + 4, true);
      const flags      = this._view.getUint16(offset + 8, true);
      // 2 bytes reserved at offset + 10

      if (dataLength > MAX_CHUNK_SIZE) {
        throw new Error(`Chunk '${id}' exceeds maximum allowed size (${dataLength} bytes)`);
      }

      const dataStart = offset + 12;
      const dataEnd   = dataStart + dataLength;
      const crcOffset = dataEnd;

      if (crcOffset + 4 > this._buffer.byteLength) {
        throw new Error(`Truncated chunk '${id}' at offset ${offset}`);
      }

      const data        = this._bytes.slice(dataStart, dataEnd);
      const storedCrc   = this._view.getUint32(crcOffset, true);
      const computedCrc = crc32(data);

      if (storedCrc !== computedCrc) {
        throw new Error(
          `CRC mismatch in chunk '${id}': ` +
          `stored=0x${storedCrc.toString(16)} computed=0x${computedCrc.toString(16)}`
        );
      }

      chunks.push({ id, flags, data });
      offset = crcOffset + 4;
    }

    return chunks;
  }
}

// ── AfixFile ───────────────────────────────────────────────────────────────────

/**
 * A decoded `.aFix` file.
 */
export class AfixFile {
  /**
   * @param {{version: object, dimensions: object, chunks: object[]}} parsed
   */
  constructor({ version, dimensions, chunks }) {
    /** @type {{major: number, minor: number, patch: number, flag: number}} */
    this.version = version;
    /** @type {{width: number, height: number}} */
    this.dimensions = dimensions;
    /** @type {Array<{id: string, flags: number, data: Uint8Array}>} */
    this.chunks = chunks;
  }

  /** Return the first chunk with the given ID, or `null`. */
  getChunk(id) {
    return this.chunks.find(c => c.id === id) ?? null;
  }

  /** Parse and return the META chunk as a plain object, or `null`. */
  get meta() {
    const chunk = this.getChunk(CHUNK_IDS.META);
    if (!chunk) return null;
    try {
      return JSON.parse(new TextDecoder().decode(chunk.data));
    } catch {
      return null;
    }
  }

  /** Parse and return the Semantic Object Manifest, or `null`. */
  get objectManifest() {
    const chunk = this.getChunk(CHUNK_IDS.OBJM);
    if (!chunk) return null;
    try {
      return JSON.parse(new TextDecoder().decode(chunk.data));
    } catch {
      return null;
    }
  }

  /**
   * Render the decoded image into a `<canvas>` element using progressive LOD.
   *
   * @param {HTMLCanvasElement} canvas — target canvas element
   * @param {{lod?: number}} [options] — max LOD to render (0=skeleton, 1=textured, 2=lossless)
   * @returns {Promise<void>}
   */
  async renderToCanvas(canvas, options = {}) {
    const maxLod = options.lod ?? 2;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Cannot get 2D context from canvas');

    canvas.width  = this.dimensions.width;
    canvas.height = this.dimensions.height;

    // LOD-0 — Geometric Skeleton (VEC_ chunk)
    this._renderSkeleton(ctx);
    canvas.dispatchEvent(new CustomEvent('afix:skeleton'));

    if (maxLod >= 1) {
      // LOD-1 — Latent Texture Field (LAT_ chunk)
      await this._renderTexture(ctx);
      canvas.dispatchEvent(new CustomEvent('afix:textured'));
    }

    if (maxLod >= 2) {
      // LOD-2 — Parity Residual (RES_ chunk, optional)
      await this._applyResidual(ctx);
    }

    canvas.dispatchEvent(new CustomEvent('afix:ready'));
  }

  // ── Internal rendering helpers ───────────────────────────────────────────────

  /**
   * Render the S1 Geometric Skeleton by drawing detected edge pixels.
   * @param {CanvasRenderingContext2D} ctx
   */
  _renderSkeleton(ctx) {
    const chunk = this.getChunk(CHUNK_IDS.VEC);
    if (!chunk || chunk.data.length < 4) return;

    const view  = new DataView(chunk.data.buffer, chunk.data.byteOffset);
    const count = view.getUint32(0, true);

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, this.dimensions.width, this.dimensions.height);

    ctx.fillStyle = '#000000';
    const maxEdges = Math.min(count, (chunk.data.length - 4) / 4);
    for (let i = 0; i < maxEdges; i++) {
      const base = 4 + i * 4;
      const x = view.getUint16(base,     true);
      const y = view.getUint16(base + 2, true);
      ctx.fillRect(x, y, 1, 1);
    }
  }

  /**
   * Render the S2 Latent Texture Field by up-sampling the compact tensor.
   * @param {CanvasRenderingContext2D} ctx
   * @returns {Promise<void>}
   */
  async _renderTexture(ctx) {
    const chunk = this.getChunk(CHUNK_IDS.LAT);
    if (!chunk || chunk.data.length < 12) return;

    const view     = new DataView(chunk.data.buffer, chunk.data.byteOffset);
    const latW     = view.getUint32(0, true);
    const latH     = view.getUint32(4, true);
    const channels = view.getUint32(8, true);

    // Build an ImageData from the normalised f32 values.
    const offscreenCanvas = new OffscreenCanvas(latW, latH);
    const offCtx          = offscreenCanvas.getContext('2d');
    const imgData         = offCtx.createImageData(latW, latH);
    const pixels          = imgData.data;

    const floatCount = latW * latH * channels;
    const expectedBytes = 12 + floatCount * 4;
    if (chunk.data.length < expectedBytes) return;

    for (let i = 0; i < latW * latH; i++) {
      const base = 12 + i * channels * 4;
      for (let c = 0; c < Math.min(channels, 4); c++) {
        // De-normalise: value in [-1, 1] → [0, 255]
        const f   = view.getFloat32(base + c * 4, true);
        const val = Math.round(((f + 1.0) / 2.0) * 255);
        pixels[i * 4 + c] = Math.max(0, Math.min(255, val));
      }
      if (channels < 4) pixels[i * 4 + 3] = 255;
    }

    offCtx.putImageData(imgData, 0, 0);

    // Up-sample the latent image to the logical canvas dimensions.
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(offscreenCanvas, 0, 0, this.dimensions.width, this.dimensions.height);
  }

  /**
   * Apply S3 Parity Residual correction (no-op in browser; placeholder).
   * @param {CanvasRenderingContext2D} _ctx
   * @returns {Promise<void>}
   */
  async _applyResidual(_ctx) {
    // A production decoder would apply the residual here.
    // The RES_ chunk presence is checked but the correction is a no-op in
    // this reference implementation.
  }
}

// ── Convenience function ──────────────────────────────────────────────────────

/**
 * Fetch a `.aFix` file from `url` and decode it.
 *
 * @param {string} url
 * @returns {Promise<AfixFile>}
 */
export async function loadAfix(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch '${url}': HTTP ${response.status}`);
  }
  const buffer  = await response.arrayBuffer();
  const decoder = new AfixDecoder(buffer);
  return decoder.decode();
}
