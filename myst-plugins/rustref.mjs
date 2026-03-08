import { readdirSync, readFileSync, statSync } from 'fs';
import { basename, join } from 'path';

function grepLines(lines, searchString) {
  for (const [i, line] of lines.entries()) {
    if (line.includes(searchString)) {
      return i;
    }
  }
  return null;
}

function grepOne(dir, searchString) {
  let result = null;

  function walk(current) {
    for (const entry of readdirSync(current)) {
      const fullPath = join(current, entry);
      if (statSync(fullPath).isDirectory()) {
        walk(fullPath);
      } else {
        const lines = readFileSync(fullPath, 'utf8').split('\n');
        const found = grepLines(lines, searchString);
        if (found) {
          if (result !== null) {
            throw new Error(`Expected 1 match but found more than one for "${searchString}"`);
          }
          result = { filePath: fullPath, fileLines: lines, line: found };
        }
      }
    }
  }

  walk(dir);

  if (!result) {
    throw new Error(`No match found for "${searchString}" in ${dir}`);
  }
  return result;
}

function unindent(lines) {
  const minIndent = Math.min(
    ...lines
      .filter(line => line.trim() !== '')
      .map(line => line.match(/^(\s*)/)[1].length)
  );

  return lines.map(line => line.slice(minIndent));
}

function render(cwd, blockDesc) {
  const startMarker = `MYSTMD::${blockDesc} START`;
  const endMarker = `MYSTMD::${blockDesc} END`;

  const { filePath, fileLines, line: startLine } = grepOne(cwd, startMarker);
  const fileFromStartMarker = fileLines.slice(startLine + 1);

  if (grepLines(fileFromStartMarker, startMarker) != null) {
    throw new Error(`found multiple "${startMarker}" markers in ${filePath}`);
  }
  const endLine = grepLines(fileFromStartMarker, endMarker);
  if (endLine == null) {
    throw new Error(`couldn't find "${endMarker}" in ${filePath}`);
  }
  if (grepLines(fileFromStartMarker.slice(endLine + 1), endMarker) != null) {
    throw new Error(`found multiple "${endMarker}" markers in ${filePath}`);
  }

  const matchedLines = unindent(fileLines.slice(startLine + 1, startLine + endLine + 1));

  return [
    {
      type: 'div',
      class: 'rustref',
      children: [
        {
          type: 'code',
          lang: 'rust',
          filename: '\u200B',
          value: matchedLines.join('\n'),
        },
        {
          type: 'div',
          class: 'filelink text-sm myst-code-filename-title',
          children: [
            {
              type: 'link',
              url: 'https://github.com/yshavit/llm-notes',
              children: [
                {
                  type: 'inlinecode',
                  // +2 on start, for 0-indexing and the marker
                  // +1 on end, for 0-indexing
                  value: `${basename(filePath)}:${startLine + 2}-${startLine + endLine + 1}`,
                }
              ]
            }
          ]
        }
      ]
    },
  ];
}

const directive = {
  name: 'rustref',
  doc: 'Rust code reference',
  arg: {
    type: String,
    doc: 'Rust block description',
  },
  run(data, vfile) {
    return render(join(vfile.cwd, '..', 'simpllm', 'simpllm-core'), data.arg);
  },
};

const plugin = {
  name: 'rustref',
  directives: [directive],
  roles: [],
};

export default plugin;
