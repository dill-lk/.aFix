// Package libafix provides types and functions for reading and writing .aFix
// (Adaptive Flexible Image X) files.
//
// # File Format Overview
//
//	┌────────────────────────────────┐
//	│  HEADER  (33 bytes)            │
//	│    Magic "AFIXK" (5 B)         │
//	│    Version VSN_ (4 B)          │
//	│    Dimensions DESC (24 B)      │
//	├────────────────────────────────┤
//	│  ATOM_MAP (144 bytes)          │
//	│    6 × 24-byte chunk pointers  │
//	├────────────────────────────────┤
//	│  PAYLOAD (variable)            │
//	│    Atom chunks (see ChunkID)   │
//	└────────────────────────────────┘
package libafix

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"math"
)

// Magic is the five-byte sequence that opens every .aFix file ("AFIXK").
var Magic = [5]byte{'A', 'F', 'I', 'X', 'K'}

// AtomMapOffset is the byte offset where the ATOM_MAP begins (0x21 = 33).
const AtomMapOffset = 0x21

// AtomMapSize is the fixed size of the ATOM_MAP in bytes (144 B = 6 × 24 B).
const AtomMapSize = 144

// PayloadOffset is the byte offset where the PAYLOAD begins (0xB1 = 177).
const PayloadOffset = 0xB1

// AtomEntrySize is the size of a single ATOM_MAP entry in bytes (24 B).
const AtomEntrySize = 24

// MaxAtomEntries is the maximum number of entries in the ATOM_MAP.
const MaxAtomEntries = 6

// MaxChunkSize is the maximum allowed size of a single chunk (512 MiB).
const MaxChunkSize = 512 * 1024 * 1024

// ── ChunkID ───────────────────────────────────────────────────────────────────

// ChunkID is a registered or custom four-byte chunk identifier.
type ChunkID [4]byte

var (
	ChunkMeta        = ChunkID{'M', 'E', 'T', 'A'}
	ChunkVec         = ChunkID{'V', 'E', 'C', '_'}
	ChunkLat         = ChunkID{'L', 'A', 'T', '_'}
	ChunkRes         = ChunkID{'R', 'E', 'S', '_'}
	ChunkDepth       = ChunkID{'D', 'P', 'T', 'H'}
	ChunkSig         = ChunkID{'S', 'I', 'G', 'B'}
	ChunkObjManifest = ChunkID{'O', 'B', 'J', 'M'}
	ChunkPreview     = ChunkID{'P', 'R', 'E', 'V'}
)

// String returns the human-readable four-character name of the chunk.
func (id ChunkID) String() string {
	return string(id[:])
}

// ── Chunk ─────────────────────────────────────────────────────────────────────

// Chunk is a single atom chunk inside the .aFix PAYLOAD.
type Chunk struct {
	// ID is the four-byte chunk identifier.
	ID ChunkID
	// Flags holds per-chunk flags (bit 0 = AES-256-GCM encrypted).
	Flags uint16
	// Data is the raw chunk payload, already CRC-32 validated on read.
	Data []byte
}

// IsEncrypted returns true when flag bit 0 is set (AES-256-GCM encryption).
func (c *Chunk) IsEncrypted() bool {
	return c.Flags&0x0001 != 0
}

// ── Version ───────────────────────────────────────────────────────────────────

// Version is the protocol version packed into four bytes (MAJOR.MINOR.PATCH.FLAG).
type Version struct {
	Major uint8
	Minor uint8
	Patch uint8
	// Flag is reserved for extension flags (compression envelope, encryption).
	Flag uint8
}

// CurrentVersion returns the current protocol version (1.0.4, no flags).
func CurrentVersion() Version {
	return Version{Major: 1, Minor: 0, Patch: 4, Flag: 0}
}

// String formats the version as "MAJOR.MINOR.PATCH".
func (v Version) String() string {
	return fmt.Sprintf("%d.%d.%d", v.Major, v.Minor, v.Patch)
}

// ── Dimensions ────────────────────────────────────────────────────────────────

// Dimensions stores the logical (resolution-independent) image size.
type Dimensions struct {
	Width  float64
	Height float64
}

// ── Header ────────────────────────────────────────────────────────────────────

// Header is the .aFix file header (33 bytes, before the ATOM_MAP).
type Header struct {
	Version    Version
	Dimensions Dimensions
}

// ── AfixFile ──────────────────────────────────────────────────────────────────

// AfixFile is a parsed .aFix file containing a header and a list of atom chunks.
type AfixFile struct {
	Header Header
	Chunks []Chunk
}

// New creates an empty AfixFile with the given dimensions.
func New(width, height float64) *AfixFile {
	return &AfixFile{
		Header: Header{
			Version:    CurrentVersion(),
			Dimensions: Dimensions{Width: width, Height: height},
		},
	}
}

// AddChunk appends a chunk to the file.
func (f *AfixFile) AddChunk(c Chunk) {
	f.Chunks = append(f.Chunks, c)
}

// GetChunk returns the first chunk with the given ID, or nil if not present.
func (f *AfixFile) GetChunk(id ChunkID) *Chunk {
	for i := range f.Chunks {
		if f.Chunks[i].ID == id {
			return &f.Chunks[i]
		}
	}
	return nil
}

// ── Read ──────────────────────────────────────────────────────────────────────

// Read parses an .aFix file from r.
func Read(r io.ReadSeeker) (*AfixFile, error) {
	// 1. Magic
	var magic [5]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, fmt.Errorf("reading magic: %w", err)
	}
	if magic != Magic {
		return nil, fmt.Errorf("invalid magic bytes: %02X %02X %02X %02X %02X",
			magic[0], magic[1], magic[2], magic[3], magic[4])
	}

	// 2. Version (4 B)
	var vsnBuf [4]byte
	if _, err := io.ReadFull(r, vsnBuf[:]); err != nil {
		return nil, fmt.Errorf("reading version: %w", err)
	}
	version := Version{
		Major: vsnBuf[0],
		Minor: vsnBuf[1],
		Patch: vsnBuf[2],
		Flag:  vsnBuf[3],
	}

	// 3. DESC — Dimensions (24 B: two float64 LE + 8 B reserved)
	var descBuf [24]byte
	if _, err := io.ReadFull(r, descBuf[:]); err != nil {
		return nil, fmt.Errorf("reading DESC: %w", err)
	}
	width := math.Float64frombits(binary.LittleEndian.Uint64(descBuf[0:8]))
	height := math.Float64frombits(binary.LittleEndian.Uint64(descBuf[8:16]))
	// bytes 16–23 are reserved

	header := Header{
		Version:    version,
		Dimensions: Dimensions{Width: width, Height: height},
	}

	// 4. ATOM_MAP (144 B) — skip; chunks are read sequentially below
	var atomMapBuf [AtomMapSize]byte
	if _, err := io.ReadFull(r, atomMapBuf[:]); err != nil {
		return nil, fmt.Errorf("reading ATOM_MAP: %w", err)
	}

	// 5. PAYLOAD — seek to start and read chunks sequentially
	if _, err := r.Seek(PayloadOffset, io.SeekStart); err != nil {
		return nil, fmt.Errorf("seeking to payload: %w", err)
	}

	var chunks []Chunk
	for {
		// Try to read chunk ID (4 B). EOF here is normal end-of-file.
		var idBuf [4]byte
		if _, err := io.ReadFull(r, idBuf[:]); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				break
			}
			return nil, fmt.Errorf("reading chunk ID: %w", err)
		}

		var lenBuf [4]byte
		if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
			return nil, fmt.Errorf("reading chunk length: %w", err)
		}
		chunkLen := int(binary.LittleEndian.Uint32(lenBuf[:]))

		var flagsBuf [2]byte
		if _, err := io.ReadFull(r, flagsBuf[:]); err != nil {
			return nil, fmt.Errorf("reading chunk flags: %w", err)
		}
		flags := binary.LittleEndian.Uint16(flagsBuf[:])

		// Skip reserved 2 bytes
		if _, err := io.ReadFull(r, make([]byte, 2)); err != nil {
			return nil, fmt.Errorf("reading chunk reserved: %w", err)
		}

		// Safety: reject oversized chunks
		if chunkLen > MaxChunkSize {
			return nil, fmt.Errorf("chunk '%s' too large: %d bytes", ChunkID(idBuf), chunkLen)
		}

		data := make([]byte, chunkLen)
		if _, err := io.ReadFull(r, data); err != nil {
			return nil, fmt.Errorf("reading chunk data: %w", err)
		}

		var crcBuf [4]byte
		if _, err := io.ReadFull(r, crcBuf[:]); err != nil {
			return nil, fmt.Errorf("reading chunk CRC: %w", err)
		}
		storedCRC := binary.LittleEndian.Uint32(crcBuf[:])
		computedCRC := crc32IEEE(data)
		if storedCRC != computedCRC {
			return nil, fmt.Errorf("CRC mismatch in chunk '%s': stored=%#010x computed=%#010x",
				ChunkID(idBuf), storedCRC, computedCRC)
		}

		chunks = append(chunks, Chunk{
			ID:    ChunkID(idBuf),
			Flags: flags,
			Data:  data,
		})
	}

	return &AfixFile{Header: header, Chunks: chunks}, nil
}

// ── Write ─────────────────────────────────────────────────────────────────────

// Write serialises the AfixFile to w.
func (f *AfixFile) Write(w io.WriteSeeker) error {
	// 1. Magic
	if _, err := w.Write(Magic[:]); err != nil {
		return err
	}

	// 2. Version (4 B)
	v := f.Header.Version
	if _, err := w.Write([]byte{v.Major, v.Minor, v.Patch, v.Flag}); err != nil {
		return err
	}

	// 3. DESC (24 B): two float64 LE + 8 B reserved
	var descBuf [24]byte
	binary.LittleEndian.PutUint64(descBuf[0:8], math.Float64bits(f.Header.Dimensions.Width))
	binary.LittleEndian.PutUint64(descBuf[8:16], math.Float64bits(f.Header.Dimensions.Height))
	// bytes 16–23 remain zero (reserved)
	if _, err := w.Write(descBuf[:]); err != nil {
		return err
	}

	// 4. ATOM_MAP (144 B)
	atomMap := make([]byte, AtomMapSize)
	currentOffset := uint64(PayloadOffset)
	for i, chunk := range f.Chunks {
		if i >= MaxAtomEntries {
			break
		}
		base := i * AtomEntrySize
		copy(atomMap[base:base+4], chunk.ID[:])
		binary.LittleEndian.PutUint64(atomMap[base+4:base+12], currentOffset)
		binary.LittleEndian.PutUint64(atomMap[base+12:base+20], uint64(len(chunk.Data)))
		binary.LittleEndian.PutUint32(atomMap[base+20:base+24], crc32IEEE(chunk.Data))
		// Each payload chunk: 4 (id) + 4 (len) + 2 (flags) + 2 (res) + data + 4 (crc)
		currentOffset += uint64(4 + 4 + 2 + 2 + len(chunk.Data) + 4)
	}
	if _, err := w.Write(atomMap); err != nil {
		return err
	}

	// 5. PAYLOAD — write each chunk
	for _, chunk := range f.Chunks {
		// Chunk ID (4 B)
		if _, err := w.Write(chunk.ID[:]); err != nil {
			return err
		}
		// Chunk length (4 B LE)
		var lenBuf [4]byte
		binary.LittleEndian.PutUint32(lenBuf[:], uint32(len(chunk.Data)))
		if _, err := w.Write(lenBuf[:]); err != nil {
			return err
		}
		// Flags (2 B LE)
		var flagsBuf [2]byte
		binary.LittleEndian.PutUint16(flagsBuf[:], chunk.Flags)
		if _, err := w.Write(flagsBuf[:]); err != nil {
			return err
		}
		// Reserved (2 B)
		if _, err := w.Write([]byte{0, 0}); err != nil {
			return err
		}
		// Data
		if _, err := w.Write(chunk.Data); err != nil {
			return err
		}
		// CRC-32 (4 B LE)
		var crcBuf [4]byte
		binary.LittleEndian.PutUint32(crcBuf[:], crc32IEEE(chunk.Data))
		if _, err := w.Write(crcBuf[:]); err != nil {
			return err
		}
	}

	return nil
}

// ── Internal helpers ──────────────────────────────────────────────────────────

// crc32IEEE computes the CRC-32 (IEEE polynomial) of data, matching the Rust
// crc32fast crate which also uses the IEEE polynomial.
func crc32IEEE(data []byte) uint32 {
	return crc32.ChecksumIEEE(data)
}
