import fs from 'node:fs/promises';
import { Workbook } from '@oai/artifact-tool';

const output = '/Users/jeffreychen/Desktop/College/Computer Science/5106 Multi-Agent Systems/project/CSE-5106-FHIR-Multi-Agent-System/results/v3/development';
const tables = JSON.parse(await fs.readFile('/private/tmp/v3-development-consolidation/tables.json', 'utf8'));
const checks = [];
const csvValue = value => value === null || value === undefined ? '' :
  /[",\r\n]/.test(String(value)) ? '"' + String(value).replaceAll('"', '""') + '"' : String(value);

for (const [name, records] of Object.entries(tables)) {
  const headers = [...new Set(records.flatMap(row => Object.keys(row)))];
  const matrix = [headers, ...records.map(row => headers.map(key => row[key] ?? null))];
  if (matrix.some(row => row.some(value => value !== null && !['string', 'number', 'boolean'].includes(typeof value))))
    throw new Error('Only flat typed numeric/identifier data may enter CSV');
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add('Data');
  sheet.getRangeByIndexes(0, 0, matrix.length, headers.length).values = matrix;
  sheet.getRangeByIndexes(0, 0, matrix.length, headers.length).format.font = { name: 'Arial', size: 10 };
  sheet.getRangeByIndexes(0, 0, 1, headers.length).format = {
    fill: '#243746', font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' },
  };
  sheet.freezePanes.freezeRows(1);
  sheet.getRangeByIndexes(0, 0, matrix.length, headers.length).format.columnWidth = 23;
  sheet.getRangeByIndexes(0, 0, 1, headers.length).format.rowHeight = 34;
  const identifierColumn = headers.indexOf('question_id');
  if (identifierColumn >= 0) sheet.getRangeByIndexes(1, identifierColumn, records.length, 1).setNumberFormat('@');
  workbook.recalculate();
  const observed = sheet.getUsedRange().values;
  if (JSON.stringify(observed) !== JSON.stringify(matrix)) throw new Error('Typed data matrix changed during spreadsheet authoring');
  const check = await workbook.inspect({ kind: 'table', range: 'Data!A1:H5', tableMaxRows: 5, tableMaxCols: 8, maxChars: 1200 });
  const errors = await workbook.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!', options: { useRegex: true, maxResults: 20 }, maxChars: 1000 });
  const preview = await workbook.render({ sheetName: 'Data', range: `A1:${String.fromCharCode(64 + Math.min(8, headers.length))}6`, scale: 1.2, format: 'png' });
  await fs.writeFile(`/private/tmp/v3-development-consolidation/${name}.png`, new Uint8Array(await preview.arrayBuffer()));
  await fs.writeFile(`${output}/${name}.csv`, observed.map(row => row.map(csvValue).join(',')).join('\r\n') + '\r\n');
  checks.push({ name, rows: records.length, columns: headers.length, exactTypedMatrix: true, inspection: check.ndjson, formulaScan: errors.ndjson });
}
await fs.writeFile('/private/tmp/v3-development-consolidation/csv_validation.json', JSON.stringify(checks, null, 2) + '\n');
console.log(JSON.stringify(checks.map(({name,rows,columns,exactTypedMatrix}) => ({name,rows,columns,exactTypedMatrix}))));
