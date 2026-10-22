"""Checks the five datasets are in data/ and are the exact files we used (SHA-256)."""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
EXPECTED = {
    'hotel_bookings.csv': '7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06',
    'KaggleV2-May-2016.csv': '9132d3e7d0246617df9041d3764f20ad6f08e7b0d9f0997fa254fc5e52eda27d',
    'occupancy+detection/datatraining.txt': 'b2c4d0ce2b9e4e453c476f7125ef31aeec2d1f5c7f5572d0e80de3df6521ab56',
    'occupancy+detection/datatest.txt': '1b92c7c1b2838963464fa891a610cf3c5db4becb7189189b29b330107a584c7f',
    'occupancy+detection/datatest2.txt': 'd026d1bd5aeccd4aff4f3b3710d48e40613bd5fc370db7e61bbdcaa50d985095',
    'room_occupancy_estimation/Occupancy_Estimation.csv': 'c090ee5c94b61762bfec9d22767864490b51030f98cca030f02822468db25d9c',
    'robod/combined_Room1.csv': '7275ff537f6e15fa2a96e6d071c56cdab802e1300a5713543dda0a45c659a197',
    'robod/combined_Room2.csv': '5d604bbe9184666d31cf3206dc85589796e8ac9d77d851bb08a3d45879c09504',
    'robod/combined_Room3.csv': 'ca76b5832061637010c8a516f077e47dfb813b017b2e63c2fd8b2d1f937a1aa7',
    'robod/combined_Room4.csv': 'bd30d0045410403d218528cbb9bdaa6b92acfade2a2ee8ce15463766aedf4110',
    'robod/combined_Room5.csv': 'dc33b443a78d3e6c874868616187b6444db73000d54f546ba716aba2d48f16bf',
}
ok = True
for name, digest in EXPECTED.items():
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print(f'MISSING  data/{name}  (see data/README.md)')
        ok = False
        continue
    actual = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    status = 'OK      ' if actual == digest else 'CHANGED '
    ok &= actual == digest
    print(f'{status} data/{name}')
sys.exit(0 if ok else 1)
