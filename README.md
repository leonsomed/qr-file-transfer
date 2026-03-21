Scan static QR code:

```bash
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
python qr_reader.py -o ./scan.json
```

Send a file: 

```bash
python qr_send_file.py /path/to/file [--chunk-chars N] [--dwell SEC]
```

Receive a file:

```bash
python qr_receive_file.py [--camera 0] [--output-dir .]
```