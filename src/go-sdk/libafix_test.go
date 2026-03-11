package libafix_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"strings"
	"testing"

	libafix "github.com/dill-lk/afix-go"
)

// makeSimpleFile builds a small AfixFile with META and VEC_ chunks, mirroring
// the Rust make_simple_file() helper in lib.rs tests.
func makeSimpleFile() *libafix.AfixFile {
	f := libafix.New(1920, 1080)
	f.AddChunk(libafix.Chunk{
		ID:   libafix.ChunkMeta,
		Data: []byte(`{"version":"1.0","creator":"test"}`),
	})
	f.AddChunk(libafix.Chunk{
		ID:   libafix.ChunkVec,
		Data: []byte{0xDE, 0xAD, 0xBE, 0xEF},
	})
	return f
}

// writeRead is a helper that writes f to a buffer and reads it back.
func writeRead(t *testing.T, f *libafix.AfixFile) *libafix.AfixFile {
	t.Helper()
	buf := new(bytes.Buffer)
	rw := &readWriter{buf: buf}
	if err := f.Write(rw); err != nil {
		t.Fatalf("Write: %v", err)
	}
	rw.pos = 0
	got, err := libafix.Read(rw)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	return got
}

func TestRoundtripWriteRead(t *testing.T) {
	original := makeSimpleFile()
	parsed := writeRead(t, original)

	if parsed.Header.Dimensions.Width != 1920 {
		t.Errorf("width: got %v, want 1920", parsed.Header.Dimensions.Width)
	}
	if parsed.Header.Dimensions.Height != 1080 {
		t.Errorf("height: got %v, want 1080", parsed.Header.Dimensions.Height)
	}
	if len(parsed.Chunks) != 2 {
		t.Fatalf("chunk count: got %d, want 2", len(parsed.Chunks))
	}
	if parsed.Chunks[0].ID != libafix.ChunkMeta {
		t.Errorf("chunks[0].ID: got %v, want META", parsed.Chunks[0].ID)
	}
	if parsed.Chunks[1].ID != libafix.ChunkVec {
		t.Errorf("chunks[1].ID: got %v, want VEC_", parsed.Chunks[1].ID)
	}
}

func TestBadMagicIsRejected(t *testing.T) {
	buf := bytes.NewReader([]byte("NOPE\x00"))
	rw := &readerSeeker{r: buf}
	_, err := libafix.Read(rw)
	if err == nil {
		t.Fatal("expected error for bad magic, got nil")
	}
	if !strings.Contains(err.Error(), "invalid magic") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestCRCMismatchIsRejected(t *testing.T) {
	f := makeSimpleFile()
	buf := new(bytes.Buffer)
	rw := &readWriter{buf: buf}
	if err := f.Write(rw); err != nil {
		t.Fatalf("Write: %v", err)
	}

	// Flip a byte in the first chunk's data region.
	// PAYLOAD starts at 0xB1. First chunk header is 4+4+2+2 = 12 bytes, then data.
	raw := buf.Bytes()
	flipPos := libafix.PayloadOffset + 12
	raw[flipPos] ^= 0xFF

	rs := &readerSeeker{r: bytes.NewReader(raw)}
	_, err := libafix.Read(rs)
	if err == nil {
		t.Fatal("expected CRC mismatch error, got nil")
	}
	if !strings.Contains(err.Error(), "CRC mismatch") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestMagicBytesAreCorrect(t *testing.T) {
	want := [5]byte{'A', 'F', 'I', 'X', 'K'}
	if libafix.Magic != want {
		t.Errorf("Magic: got %v, want %v", libafix.Magic, want)
	}
}

func TestPayloadOffsetIsCorrect(t *testing.T) {
	if libafix.PayloadOffset != 0xB1 {
		t.Errorf("PayloadOffset: got %#x, want 0xB1", libafix.PayloadOffset)
	}
}

func TestVersionString(t *testing.T) {
	v := libafix.CurrentVersion()
	if v.String() != "1.0.4" {
		t.Errorf("Version.String(): got %q, want %q", v.String(), "1.0.4")
	}
}

func TestGetChunk(t *testing.T) {
	f := makeSimpleFile()
	c := f.GetChunk(libafix.ChunkMeta)
	if c == nil {
		t.Fatal("GetChunk(META): expected chunk, got nil")
	}
	if c.ID != libafix.ChunkMeta {
		t.Errorf("GetChunk(META).ID: got %v, want META", c.ID)
	}
	if f.GetChunk(libafix.ChunkLat) != nil {
		t.Error("GetChunk(LAT_): expected nil for absent chunk")
	}
}

func TestChunkIsEncrypted(t *testing.T) {
	plain := libafix.Chunk{ID: libafix.ChunkMeta, Flags: 0x0000, Data: []byte("x")}
	enc := libafix.Chunk{ID: libafix.ChunkMeta, Flags: 0x0001, Data: []byte("x")}
	if plain.IsEncrypted() {
		t.Error("plain chunk should not be encrypted")
	}
	if !enc.IsEncrypted() {
		t.Error("encrypted chunk should report IsEncrypted=true")
	}
}

func TestChunkDataPreserved(t *testing.T) {
	payload := []byte(`{"version":"1.0","creator":"test"}`)
	f := libafix.New(100, 200)
	f.AddChunk(libafix.Chunk{ID: libafix.ChunkMeta, Data: payload})
	parsed := writeRead(t, f)
	got := parsed.GetChunk(libafix.ChunkMeta)
	if got == nil {
		t.Fatal("META chunk missing after roundtrip")
	}
	if !bytes.Equal(got.Data, payload) {
		t.Errorf("META data mismatch: got %q, want %q", got.Data, payload)
	}
}

func TestAtomMapOffsetIsHeaderSize(t *testing.T) {
	// Header = 5 (magic) + 4 (version) + 24 (DESC) = 33 = 0x21
	if libafix.AtomMapOffset != 33 {
		t.Errorf("AtomMapOffset: got %d, want 33", libafix.AtomMapOffset)
	}
}

func TestWriteAtomMapHasCorrectOffset(t *testing.T) {
	// The ATOM_MAP entry for the first chunk should record PayloadOffset as its
	// byte_offset field (bytes 4–11 of the first 24-byte entry).
	f := libafix.New(64, 64)
	f.AddChunk(libafix.Chunk{ID: libafix.ChunkMeta, Data: []byte("hello")})

	buf := new(bytes.Buffer)
	rw := &readWriter{buf: buf}
	if err := f.Write(rw); err != nil {
		t.Fatalf("Write: %v", err)
	}
	raw := buf.Bytes()

	// ATOM_MAP starts at AtomMapOffset
	entryOffset := binary.LittleEndian.Uint64(raw[libafix.AtomMapOffset+4 : libafix.AtomMapOffset+12])
	if entryOffset != uint64(libafix.PayloadOffset) {
		t.Errorf("atom map byte_offset: got %d, want %d", entryOffset, libafix.PayloadOffset)
	}
}

// ── I/O helpers ───────────────────────────────────────────────────────────────

// readWriter wraps *bytes.Buffer to satisfy io.WriteSeeker for Write
// and io.ReadSeeker for Read by tracking position manually.
type readWriter struct {
	buf *bytes.Buffer
	pos int
}

func (rw *readWriter) Write(p []byte) (int, error) {
	n, err := rw.buf.Write(p)
	rw.pos += n
	return n, err
}

func (rw *readWriter) Seek(offset int64, whence int) (int64, error) {
	var newPos int64
	switch whence {
	case io.SeekStart:
		newPos = offset
	case io.SeekCurrent:
		newPos = int64(rw.pos) + offset
	case io.SeekEnd:
		newPos = int64(rw.buf.Len()) + offset
	default:
		return 0, fmt.Errorf("invalid whence value: %d", whence)
	}
	if newPos < 0 {
		return 0, fmt.Errorf("seek to negative position: %d", newPos)
	}
	rw.pos = int(newPos)
	return newPos, nil
}

func (rw *readWriter) Read(p []byte) (int, error) {
	return rw.buf.Read(p)
}

// readerSeeker wraps *bytes.Reader to satisfy io.ReadSeeker.
type readerSeeker struct {
	r io.ReadSeeker
}

func (rs *readerSeeker) Read(p []byte) (int, error) { return rs.r.Read(p) }
func (rs *readerSeeker) Seek(offset int64, whence int) (int64, error) {
	return rs.r.Seek(offset, whence)
}
