qr-file-transfer is a utility to transfer files between devices via an ad-hoc protocol with QR codes and cameras. It support several modes:

- static
- one-way
- bi-directional

This project was created to transfer files to an arigap device. However, it is also useful to transfer files to systems that might be compromised and you would prefer not to use storage devices like USB sticks. Also, it is useful in case you don't have a USB stick and the devices do not support any other form file transfer.

Keep in mind that transfer speed via this protocol is very slow compared to most transfer protocols. If you expect to transfer more than a few MBs it would take a very long time so you might want to find a different way. However, it is very reasonable to transfer files below a few MBs.

### Static mode

This is the simplest of them all just scan a QR code and save the string content to a file. Mainly intented to be used to scan selfcrypt QR codes.

### One-way mode

A sender script takes a file and transform bytes to a hex string. Then splits the string into chunks. Encode each chunk into a QR code. Display QR codes one at a time in quick succession and loop over the list of chunks indefinetly. Separately, a receiver scans the QR codes and assembles them together to build the original file. This is inteded for a few scenarios:

- transfering small files (a couple KBs at most)
- one of sender or receiver device has no camera for bi-directional mode
- receiver device is fully airgapped and it is not acceptable to risk data leaks in case of system compromise (extremely rare, but technically possible)

### Bi-directional mode

Similar to one-way, but receiver is able to send QR codes to sender to manage control flow. Receiver can specify missing chunks to avoid sender looping over the whole list of chunks. This greately reduces the transfer time for large files and it should be used whenever transfer speed is the priority and both devices have a camera and a screen.

## Setup

Run the following with python 3:

```bash
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```

## Getting started

### Scan static QR code:

```bash
python qr_reader.py -o ./scan.json
```

### One-way file transfer (default)

```bash
python qr_send_file.py /path/to/file [--chunk-chars N] [--dwell SEC]
python qr_receive_file.py [--camera 0] [--output-dir .]
```

### Bidirectional file transfer

Both sides use a **camera** and a **screen**. The sender must see the receiver’s display (and vice versa). The sender runs with `--mode bidirectional` and a camera index; the receiver runs with `--mode bidirectional`. Flow:

1. Sender shows `file_metadata` until its camera sees the **same** metadata JSON echoed on the receiver’s screen.
2. Sender sends `data_chunk` QRs; each chunk JSON includes `sha256` so the receiver can ignore foreign payloads.
3. The receiver shows a **split view**: camera feed on the left, outbound QR on the right (metadata echo, then `missing_ranges` control). Incoming QRs are decoded from the camera frame.
4. After the first control QR (post-handshake), the receiver refreshes `missing_ranges` only after **three** duplicate decodes of **three different chunk orders** in a row (several decodes of the same order in a row count as one). Any non-duplicate decode resets the counter.
5. Optional **resume cache** (bidirectional only): `--progress-cache PATH` writes progress every 60s (override with `--progress-interval`). On restart, state is loaded and `missing_ranges` is shown immediately. `--progress-cache` is rejected if `--mode` is not `bidirectional`.

```bash
python qr_send_file.py /path/to/file --mode bidirectional --camera 0
python qr_receive_file.py --mode bidirectional --camera 0 --output-dir . [--progress-cache ./qr-progress.json]
```

Use different `--camera` values if both apps run on one machine, or two machines pointed at each other’s screens.
