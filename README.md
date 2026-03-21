Scan static QR code:

```bash
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
python qr_reader.py -o ./scan.json
```

### One-way file transfer (default)

Sender shows QR codes only; receiver uses the camera.

```bash
python qr_send_file.py /path/to/file [--chunk-chars N] [--dwell SEC]
python qr_receive_file.py [--camera 0] [--output-dir .]
```

### Bidirectional file transfer

Both sides use a **camera** and a **screen**. The sender must see the receiver’s display (and vice versa). The sender runs with `--mode bidirectional` and a camera index; the receiver runs with `--mode bidirectional`. Flow:

1. Sender shows `file_metadata` until its camera sees the **same** metadata JSON echoed on the receiver’s screen.
2. Sender sends `data_chunk` QRs; each chunk JSON includes `sha256` so the receiver can ignore foreign payloads.
3. The receiver shows a **split view**: camera feed on the left (decode ROI), outbound QR on the right (metadata echo, then `missing_ranges` control).
4. If the sender repeats chunks or loops before the file is complete, the receiver refreshes `missing_ranges` only after a **5s confirmation** window with continued redundant/loop signals.
5. Optional **resume cache** (bidirectional only): `--progress-cache PATH` writes progress every 60s (override with `--progress-interval`). On restart, state is loaded and `missing_ranges` is shown immediately. `--progress-cache` is rejected if `--mode` is not `bidirectional`.

```bash
python qr_send_file.py /path/to/file --mode bidirectional --camera 0
python qr_receive_file.py --mode bidirectional --camera 0 --output-dir . \
  [--progress-cache ./qr-progress.json]
```

Use different `--camera` values if both apps run on one machine, or two machines pointed at each other’s screens.
