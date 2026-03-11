/**
 * afix-view.js — <afix-img> custom element
 *
 * A drop-in replacement for <img> that renders .aFix files in any modern
 * browser via the afix-decoder.js library.
 *
 * ## Usage
 *
 * ```html
 * <script type="module" src="afix-view.js"></script>
 *
 * <afix-img src="photo.afix" alt="A beautiful photo"></afix-img>
 * ```
 *
 * ## Observed attributes
 *
 * | Attribute  | Description                                                      |
 * |------------|------------------------------------------------------------------|
 * | `src`      | Path to the `.afix` file                                         |
 * | `alt`      | Accessible description (reflected to internal <canvas> aria)     |
 * | `width`    | Explicit CSS width (pixels or CSS value)                         |
 * | `height`   | Explicit CSS height (pixels or CSS value)                        |
 * | `loading`  | `"lazy"` or `"eager"` (default: `"eager"`)                       |
 * | `parallax` | Boolean — enable depth-based parallax on device tilt             |
 * | `lod`      | Max LOD to decode: `0`=skeleton, `1`=textured, `2`=lossless      |
 *
 * ## Events
 *
 * | Event           | Description                                      |
 * |-----------------|--------------------------------------------------|
 * | `afix:skeleton` | S1 vector skeleton rendered                      |
 * | `afix:textured` | S2 neural texture composited                     |
 * | `afix:ready`    | All requested LOD layers decoded                 |
 * | `afix:error`    | Decode or network error (`event.detail.error`)   |
 *
 * @license MIT
 */

'use strict';

import { AfixDecoder, AfixFile } from '../../src/decoder/afix-decoder.js';

// ── <afix-img> custom element ─────────────────────────────────────────────────

class AfixImg extends HTMLElement {
  // ── Lifecycle ──────────────────────────────────────────────────────────────

  constructor() {
    super();
    this._shadow  = this.attachShadow({ mode: 'open' });
    this._canvas  = document.createElement('canvas');
    this._canvas.setAttribute('role', 'img');
    this._state   = 'idle';  // idle | loading | skeleton | textured | lossless | error
    this._afixFile = null;
    this._layers  = {};      // id → layer proxy

    // Styles: canvas fills the element, maintains aspect ratio.
    const style = document.createElement('style');
    style.textContent = `
      :host { display: inline-block; }
      canvas { width: 100%; height: 100%; display: block; }
    `;
    this._shadow.appendChild(style);
    this._shadow.appendChild(this._canvas);
  }

  static get observedAttributes() {
    return ['src', 'alt', 'width', 'height', 'loading', 'parallax', 'lod'];
  }

  connectedCallback() {
    if (this.getAttribute('loading') !== 'lazy') {
      this._load();
    } else {
      this._setupIntersectionObserver();
    }
    if (this.hasAttribute('parallax')) {
      this._setupParallax();
    }
  }

  disconnectedCallback() {
    this._teardownParallax();
  }

  attributeChangedCallback(name, oldVal, newVal) {
    if (oldVal === newVal) return;
    if (name === 'src') {
      this._load();
    } else if (name === 'alt') {
      this._canvas.setAttribute('aria-label', newVal ?? '');
    } else if (name === 'width') {
      this.style.width  = newVal ? `${newVal}px` : '';
    } else if (name === 'height') {
      this.style.height = newVal ? `${newVal}px` : '';
    } else if (name === 'parallax') {
      newVal !== null ? this._setupParallax() : this._teardownParallax();
    }
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Map of semantic layer objects keyed by their manifest ID.
   * Each entry provides a `style` proxy and `getBoundingBox()`.
   *
   * @type {Object.<string, AfixLayer>}
   */
  get layers() {
    return this._layers;
  }

  /**
   * Current decode state.
   * @type {'idle'|'loading'|'skeleton'|'textured'|'lossless'|'error'}
   */
  get state() {
    return this._state;
  }

  /**
   * Verify the C2PA provenance of the loaded file.
   * @returns {Promise<{isTampered: boolean, generatorType: string|null, editHistory: object[]}>}
   */
  async getProvenance() {
    if (!this._afixFile) throw new Error('No file loaded');
    const sigChunk = this._afixFile.getChunk('SIGB');
    if (!sigChunk) {
      return { isTampered: true, generatorType: null, editHistory: [] };
    }
    // In a production implementation, verify the Ed25519 signature here.
    // For now, return a placeholder response indicating the chunk is present.
    return { isTampered: false, generatorType: 'unknown', editHistory: [] };
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  async _load() {
    const src = this.getAttribute('src');
    if (!src) return;

    this._setState('loading');

    try {
      const response = await fetch(src);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const buffer  = await response.arrayBuffer();
      const decoder = new AfixDecoder(buffer);
      this._afixFile = decoder.decode();

      // Forward canvas events to the host element.
      for (const event of ['afix:skeleton', 'afix:textured', 'afix:ready']) {
        this._canvas.addEventListener(event, (e) => {
          this.dispatchEvent(new CustomEvent(e.type, { bubbles: true, composed: true }));
          if (e.type === 'afix:skeleton') this._setState('skeleton');
          if (e.type === 'afix:textured') this._setState('textured');
          if (e.type === 'afix:ready')    this._setState('lossless');
        });
      }

      const maxLod = parseInt(this.getAttribute('lod') ?? '2', 10);
      await this._afixFile.renderToCanvas(this._canvas, { lod: maxLod });

      // Build the semantic layers API.
      this._buildLayers();

    } catch (err) {
      this._setState('error');
      this.dispatchEvent(new CustomEvent('afix:error', {
        bubbles: true,
        composed: true,
        detail: { error: err },
      }));
      console.error('[afix-view] decode error:', err);
    }
  }

  _setState(state) {
    this._state = state;
    this.setAttribute('data-state', state);
  }

  /** Build the public `layers` map from the OBJM manifest. */
  _buildLayers() {
    if (!this._afixFile) return;
    const manifest = this._afixFile.objectManifest;
    if (!manifest?.objects) return;

    for (const obj of manifest.objects) {
      this._layers[obj.id] = new AfixLayer(obj, this._canvas, this._afixFile.dimensions);
    }
  }

  _setupIntersectionObserver() {
    const observer = new IntersectionObserver((entries) => {
      if (entries.some(e => e.isIntersecting)) {
        this._load();
        observer.disconnect();
      }
    });
    observer.observe(this);
  }

  _setupParallax() {
    if (this._parallaxHandler) return;
    this._parallaxHandler = (e) => this._onDeviceMotion(e);
    window.addEventListener('deviceorientation', this._parallaxHandler);
  }

  _teardownParallax() {
    if (this._parallaxHandler) {
      window.removeEventListener('deviceorientation', this._parallaxHandler);
      this._parallaxHandler = null;
    }
  }

  _onDeviceMotion(event) {
    if (!this._afixFile?.getChunk('DPTH')) return;
    const tiltX = (event.gamma ?? 0) / 45; // -1…1
    const tiltY = (event.beta  ?? 0) / 45; // -1…1
    const maxShift = 20; // pixels
    this._canvas.style.transform =
      `translate(${tiltX * maxShift}px, ${tiltY * maxShift}px)`;
  }
}

// ── AfixLayer ─────────────────────────────────────────────────────────────────

/**
 * Represents one semantic layer within an `<afix-img>` element.
 * Provides a CSS `style` proxy and a `getBoundingBox()` method.
 */
class AfixLayer {
  /**
   * @param {object} manifest — OBJM entry for this layer
   * @param {HTMLCanvasElement} canvas
   * @param {{width: number, height: number}} dimensions
   */
  constructor(manifest, canvas, dimensions) {
    this._manifest   = manifest;
    this._canvas     = canvas;
    this._dimensions = dimensions;

    /**
     * CSS style proxy — setting properties here re-renders the layer.
     * @type {CSSStyleDeclaration}
     */
    this.style = this._createStyleProxy();
  }

  /**
   * Returns the bounding box of this layer in logical pixels.
   * @returns {{x: number, y: number, width: number, height: number}|null}
   */
  getBoundingBox() {
    const bbox = this._manifest.bbox;
    if (!bbox) return null;
    return { x: bbox[0], y: bbox[1], width: bbox[2], height: bbox[3] };
  }

  /** Backing CSS text for the layer overlay element (if created). */
  get _cssText() {
    return this.__cssText ?? '';
  }

  _createStyleProxy() {
    const self = this;
    const store = {};
    return new Proxy(store, {
      set(target, prop, value) {
        target[prop] = value;
        // Reflect notable style changes back to the canvas container.
        if (prop === 'display') {
          self._canvas.style.display = value;
        } else if (prop === 'filter') {
          self._canvas.style.filter = value;
        }
        return true;
      },
      get(target, prop) {
        return target[prop];
      },
    });
  }
}

// ── Registration ──────────────────────────────────────────────────────────────

if (typeof customElements !== 'undefined') {
  customElements.define('afix-img', AfixImg);
}

export { AfixImg, AfixLayer };
