import hashlib
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.table = self.row = self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.table = []
        elif tag == 'tr' and self.table is not None:
            self.row = []
        elif tag in ('td', 'th') and self.row is not None:
            self.cell = []

    def handle_data(self, text):
        if self.cell is not None:
            self.cell.append(text)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(' '.join(' '.join(self.cell).split()))
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == 'table' and self.table is not None:
            self.tables.append(self.table)
            self.table = None


target = Path('/private/tmp/fhir-azure-status-20261015-recheck')
target.mkdir(exist_ok=True)
url = 'https://azure.status.microsoft/en-us/status'
started = datetime.now(timezone.utc).isoformat()
with urlopen(Request(url, headers={'Cache-Control': 'no-cache'}), timeout=30) as response:
    content = response.read(16 * 1024 * 1024 + 1)
    if len(content) > 16 * 1024 * 1024:
        raise ValueError('Public page exceeds download bound')
    metadata = {'url': url, 'started_at_utc': started,
                'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                'http_status': response.status,
                'headers': {name: response.headers.get(name) for name in ('Date', 'Cache-Control')},
                'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
text = content.decode('utf-8')
parser = Tables()
parser.feed(text)
matches = []
for table in parser.tables:
    headers = next((row for row in table if 'East US 2' in row), None)
    if headers:
        for row in table:
            if row and row[0] == 'Azure OpenAI Service' and len(row) == len(headers):
                matches.append({'header': headers, 'row': row,
                                'East US 2': row[headers.index('East US 2')]})
metadata['openai_east_us_2_rows'] = matches
metadata['no_active_public_events_text_present'] = 'There are currently no active events.' in text
metadata['authenticated_incident_resolution_verified'] = False
(target / 'current.html').write_bytes(content)
(target / 'observations.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata))
