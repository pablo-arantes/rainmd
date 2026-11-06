/**
 * The %FLAG/%FORMAT reader for AMBER topology files.
 *
 * 🔴 THESE FIELDS ARE FIXED WIDTH AND MUST NOT BE SPLIT ON WHITESPACE. A
 * `%FORMAT(10I8)` line holding a large negative index reads
 * `-1234567-7654321`, with no space between the two numbers, and
 * `line.trim().split(/\s+/)` returns ONE token for two values - silently, with
 * the right type and a plausible magnitude, shifting every index after it by
 * one. Dihedral atom indices are stored as 3*atom and signed, so this is not a
 * hypothetical format corner: it is what a large solvated system writes. Every
 * field here is sliced by its declared width.
 *
 * 🔴 AND THE LAST LINE OF A SECTION IS SHORT. 10I8 with 23 values writes two
 * full lines and a third holding three fields, so slicing ten fields out of
 * every line and trusting the count returns seven empty strings that parse as
 * NaN. The reader takes what the line actually holds and stops on the declared
 * element count, not on the line geometry.
 */

/**
 * @typedef {object} FortranFormat
 * @property {number} perLine declared fields per line
 * @property {"integer"|"float"|"string"} kind
 * @property {number} width characters per field
 */

/**
 * Parse a `%FORMAT(...)` descriptor.
 *
 * Handles the four that AMBER writes - `10I8`, `5E16.8`, `20a4`, `1a80` - plus
 * the `F` form some older tools emit. The leading repeat count is optional in
 * the standard and present in every file seen, so it defaults to 1.
 *
 * @param {string} descriptor the text inside the parentheses, e.g. "5E16.8"
 * @returns {FortranFormat}
 */
export function parseFortranFormat(descriptor) {
  const match = /^\s*(\d*)\s*([aAiIeEfFgG])\s*(\d+)(?:\.(\d+))?\s*$/.exec(descriptor);
  if (match === null) {
    throw new Error(`unsupported AMBER %FORMAT descriptor: ${JSON.stringify(descriptor)}`);
  }
  const [, repeat, letter, width] = match;
  const lower = letter.toLowerCase();
  return {
    perLine: repeat === "" ? 1 : Number(repeat),
    kind: lower === "a" ? "string" : lower === "i" ? "integer" : "float",
    width: Number(width),
  };
}

/**
 * Split one text line into at most `perLine` fields of `width` characters.
 *
 * Trailing fields that the line does not reach are omitted rather than
 * returned empty, so a caller counting values sees the truth about a short
 * final line.
 *
 * @param {string} line
 * @param {FortranFormat} format
 * @returns {string[]}
 */
export function sliceFields(line, format) {
  const fields = [];
  for (let index = 0; index < format.perLine; index += 1) {
    const start = index * format.width;
    if (start >= line.length) break;
    fields.push(line.slice(start, start + format.width));
  }
  return fields;
}

/**
 * Read an AMBER parm7 file into its raw sections.
 *
 * Returns the text of each `%FLAG`, already converted to the declared type but
 * otherwise untouched: no unit conversion, no index arithmetic, no
 * interpretation. prmtop.js is what interprets them, and keeping the two apart
 * is what lets a parse be checked against a file by eye.
 *
 * 🔴 A `%COMMENT` LINE MAY APPEAR ANYWHERE, INCLUDING BETWEEN %FLAG AND
 * %FORMAT, and it is not data. Treating one as a data line puts its prose
 * through Number() and fills a section with NaN.
 *
 * @param {string} text the whole file
 * @returns {{version: string|null, sections: Map<string, {format: FortranFormat, values: (number|string)[]}>}}
 */
export function readAmberSections(text) {
  const lines = text.split(/\r?\n/);
  /** @type {Map<string, {format: FortranFormat, values: (number|string)[]}>} */
  const sections = new Map();
  let version = null;
  let flag = null;
  let format = null;
  /** @type {(number|string)[]} */
  let values = [];

  const commit = () => {
    if (flag !== null && format !== null) {
      sections.set(flag, { format, values });
    }
    flag = null;
    format = null;
    values = [];
  };

  for (const line of lines) {
    if (line.startsWith("%VERSION")) {
      version = line.slice("%VERSION".length).trim();
      continue;
    }
    if (line.startsWith("%COMMENT")) continue;
    if (line.startsWith("%FLAG")) {
      commit();
      flag = line.slice("%FLAG".length).trim();
      continue;
    }
    if (line.startsWith("%FORMAT")) {
      const open = line.indexOf("(");
      const close = line.lastIndexOf(")");
      if (open === -1 || close === -1 || close < open) {
        throw new Error(`malformed %FORMAT line: ${JSON.stringify(line)}`);
      }
      format = parseFortranFormat(line.slice(open + 1, close));
      continue;
    }
    if (flag === null || format === null) continue;
    for (const field of sliceFields(line, format)) {
      if (format.kind === "string") {
        values.push(field);
        continue;
      }
      const trimmed = field.trim();
      // A blank field inside a line happens where a section's element count
      // ends mid-line; it is padding, not a zero.
      if (trimmed === "") continue;
      const parsed = Number(trimmed);
      if (!Number.isFinite(parsed)) {
        throw new Error(
          `non-numeric field ${JSON.stringify(field)} in %FLAG ${flag}`,
        );
      }
      values.push(parsed);
    }
  }
  commit();
  return { version, sections };
}
