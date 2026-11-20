'use strict';
// Refuse inputs that the browser would silently round or discard.
(function(root) {
  function decimalValue(text) {
    let [base, exponent = '0'] = text.toLowerCase().split('e');
    const negative = base.startsWith('-');
    if (negative) base = base.slice(1);
    const parts = base.split('.');
    let digits = parts.join('').replace(/^0+/, '');
    if (!digits) return '0';
    if (exponent.replace(/^[+-]/, '').length > 6) throw new Error('Unsupported JSON number.');
    let power = BigInt(exponent) - BigInt((parts[1] || '').length);
    const tail = digits.match(/0+$/);
    if (tail) { power += BigInt(tail[0].length); digits = digits.slice(0, -tail[0].length); }
    return (negative ? '-' : '') + digits + 'e' + power;
  }
  function parse(text) {
    let at = 0;
    function space() { while (at < text.length && /[\x20\t\r\n]/.test(text[at])) at++; }
    function string() {
      const match = /"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"/y;
      match.lastIndex = at;
      const found = match.exec(text);
      if (!found) throw new Error('Invalid JSON string.');
      at = match.lastIndex;
      return JSON.parse(found[0]);
    }
    function value(depth) {
      if (depth > 100) throw new Error('JSON nesting exceeds 100 levels.');
      space();
      if (text[at] === '"') return string();
      if (text[at] === '{') {
        at++; space(); const object = Object.create(null), keys = new Set();
        if (text[at] === '}') { at++; return object; }
        while (true) {
          space(); const key = string(); space();
          if (keys.has(key)) throw new Error('Duplicate JSON keys are unsupported. Use unique keys before importing.');
          keys.add(key);
          if (text[at++] !== ':') throw new Error('Expected a JSON colon.');
          object[key] = value(depth + 1); space();
          if (text[at] === '}') { at++; return object; }
          if (text[at++] !== ',') throw new Error('Expected a JSON comma.');
        }
      }
      if (text[at] === '[') {
        at++; space(); const array = [];
        if (text[at] === ']') { at++; return array; }
        while (true) {
          array.push(value(depth + 1)); space();
          if (text[at] === ']') { at++; return array; }
          if (text[at++] !== ',') throw new Error('Expected a JSON comma.');
        }
      }
      for (const [word, result] of [['true',true],['false',false],['null',null]]) {
        if (text.startsWith(word,at)) { at += word.length; return result; }
      }
      const match = /-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/y;
      match.lastIndex = at;
      const found = match.exec(text);
      if (!found) throw new Error('Invalid JSON value.');
      const number = Number(found[0]);
      if (!Number.isFinite(number) || decimalValue(found[0]) !== decimalValue(String(number))) {
        throw new Error('A JSON number would lose precision. Store long identifiers or exact decimals as strings before importing.');
      }
      at = match.lastIndex; return number;
    }
    const result = value(0); space();
    if (at !== text.length) throw new Error('Unexpected text after JSON.');
    return result;
  }
  const api = {parse};
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.CasePatchJSON = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
