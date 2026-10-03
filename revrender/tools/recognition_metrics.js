/* Browser readout; fixture parity is checked against eval/recognition_score.py. */
function recognitionMetrics(annotation, totalRows) {
  const known = new Set(['outside', 'inside', 'wrong_outside', 'wrong_inside', 'missing']);
  const rows = Object.values(annotation.rows || {});
  const positive = rows.filter(r => r.reviewed && r.revision === 'yes' && r.key_region_complete);
  const keys = positive.flatMap(r => r.keys || []);
  const readable = keys.filter(k => k.readable && !['', '□'].includes((k.char || '').trim()));
  const cancelled = readable.filter(k => k.kind === 'cancelled');
  const valid = readable.filter(k => k.kind === 'valid');
  const judged = (k, t, m) => k.judgments?.[t]?.[m];
  const matched = cancelled.filter(k => ['B', 'C', 'D'].every(m => known.has(judged(k, 'deletion', m))));
  const fraction = (count, denominator) => ({count, denominator, rate: denominator ? count / denominator : null});
  const leakage = Object.fromEntries(['B', 'C', 'D'].map(m => [m,
    fraction(matched.filter(k => ['outside', 'wrong_outside'].includes(judged(k, 'deletion', m))).length, matched.length)]));
  const cutOff = {};
  for (const task of ['verbatim', 'deletion']) {
    const correct = new Set(task === 'verbatim' ? ['outside', 'inside'] : ['outside']);
    const assessed = valid.filter(k => ['yes', 'no'].includes(k.whitening_damage) && ['B', 'C'].every(m => known.has(judged(k, task, m))));
    const baseline = assessed.filter(k => correct.has(judged(k, task, 'B')));
    const broken = baseline.filter(k => !correct.has(judged(k, task, 'C')) && k.whitening_damage === 'yes');
    cutOff[task] = {...fraction(broken.length, baseline.length), assessed_valid_keys: assessed.length, unassessed_valid_keys: valid.length - assessed.length};
  }
  return {coverage: {
    total_rows: totalRows, reviewed_rows: rows.filter(r => r.reviewed).length, confirmed_positive_rows: positive.length,
    excluded_rows: rows.filter(r => r.reviewed && ['no', 'duplicate'].includes(r.revision)).length,
    uncertain_rows: rows.filter(r => r.reviewed && r.revision === 'uncertain').length,
    unreadable_keys: keys.length - readable.length, readable_cancelled_keys: cancelled.length,
    readable_valid_keys: valid.length, unmatched_cancelled_keys: cancelled.length - matched.length
  }, cancelled_ink_leakage: leakage, valid_ink_cut_off: cutOff};
}
if (typeof module !== 'undefined') module.exports = {recognitionMetrics};
