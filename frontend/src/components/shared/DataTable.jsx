function toNum(val, key) {
  if (typeof val === 'number') return val;
  if ((key === 'latency' || key === 'duration_sec') && val !== '' && val != null) return Number(val);
  return NaN;
}

function findExtremumPerColumn(rows, columns) {
  const bestByKey = {};
  columns.forEach((col) => {
    if (col.highlight === false) return;
    const isMinBetter = col.key === 'latency' || col.key === 'duration_sec';
    let best = isMinBetter ? Infinity : -Infinity;
    rows.forEach((r) => {
      const num = toNum(r[col.key], col.key);
      if (Number.isNaN(num)) return;
      if (isMinBetter ? num < best : num > best) best = num;
    });
    bestByKey[col.key] = best === Infinity || best === -Infinity ? null : best;
  });
  return bestByKey;
}

export default function DataTable({ columns, rows, title, subtitle }) {
  const bestByKey = findExtremumPerColumn(rows, columns);

  return (
    <div>
      {(title || subtitle) && (
        <div className="flex items-center justify-between mb-6">
          {title && <h3 className="text-lg font-semibold text-text-main">{title}</h3>}
          {subtitle && <span className="text-xs font-medium text-text-muted">{subtitle}</span>}
        </div>
      )}
      <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
        <table className="w-full text-left text-sm table-fixed">
          <colgroup>
            {columns.map((col, i) => (
              <col
                key={col.key}
                style={{
                  width: i === 0 ? '22%' : `${(100 - 22) / (columns.length - 1)}%`,
                }}
              />
            ))}
          </colgroup>
          <thead className="bg-slate-50 text-xs font-semibold uppercase tracking-wider text-text-muted border-b border-slate-200">
            <tr>
              {columns.map((col) => (
                <th key={col.key} className="px-4 py-3 w-0" scope="col">
                  {col.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {rows.map((row, i) => (
              <tr key={i} className="hover:bg-slate-50 transition-colors">
                {columns.map((col) => {
                  const val = row[col.key];
                  const numVal = toNum(val, col.key);
                  const bestForCol = bestByKey[col.key];
                  const isHighlight =
                    bestForCol != null && !Number.isNaN(numVal) && numVal === bestForCol;
                  return (
                    <td
                      key={col.key}
                      className={`px-4 py-3 font-medium text-text-main w-0 overflow-hidden text-ellipsis min-w-0 ${isHighlight ? 'bg-primary/25' : ''}`}
                    >
                      {col.render ? (
                        col.render(val, row)
                      ) : typeof val === 'number' ? (
                        col.percent ? `${(val * 100).toFixed(0)}%` : val.toFixed(2)
                      ) : (
                        val
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
