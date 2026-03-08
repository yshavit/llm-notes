import { execFileSync } from "node:child_process";
import { copyFileSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";

const pluginDir = '_plugin/drawio';

function tryReadFile(f) {
  try {
    return readFileSync(f, 'utf8');
  } catch (err) {
    if (err.code === 'ENOENT') {
      return undefined;
    }
    throw err;
  }
}

function sha1(data) {
  return createHash('sha1').update(data).digest('hex')
}

function renderDrawio(vfile, imagePath, theme) {
  const imagePathHash = sha1(imagePath) + `-${theme}`;

  const drawioPath = `${pluginDir}/${imagePathHash}.drawio`;
  copyFileSync(`${imagePath}.drawio`, drawioPath);

  var renderReason = 'render';

  const contentHashFile = `${pluginDir}/${imagePathHash}.sha1`;
  const expectedContentHash = tryReadFile(contentHashFile);
  const actualContentHash = sha1(readFileSync(drawioPath));
  if (expectedContentHash !== undefined) {
    if (actualContentHash == expectedContentHash) {
      renderReason = null;
    } else {
      renderReason = 'rerender';
    }
  }
  if (renderReason) {
    console.log(`need to ${renderReason} ${imagePath}-${theme}`);
    execFileSync('docker', [
      'run', '--rm', '-i',
      '-w', '/data',
      '-v', `${vfile.cwd}/${pluginDir}:/data`,
      'rlespinasse/drawio-desktop-headless',
      '-x',
      '-f', 'svg',
      '--svg-theme', theme,
      '-o', `${imagePathHash}.svg`,
      `${imagePathHash}.drawio`,
    ]);
    writeFileSync(contentHashFile, actualContentHash, 'utf8');
  }
  const svgContents = readFileSync(`${pluginDir}/${imagePathHash}.svg`).toString('base64');
  const dataFile = `${pluginDir}/${imagePathHash}.data`
  writeFileSync(dataFile, 'data:image/svg+xml;base64' + svgContents, 'utf8');
  return `/${pluginDir}/${imagePathHash}.svg`;
}

function render(vfile, imagePath, container, classString, alt) {
  const light = renderDrawio(vfile, imagePath, 'light');
  const dark = renderDrawio(vfile, imagePath, 'dark');

  return [
    {
      type: container,
      class: "hidden dark:block",
      children: [{
        type: "image",
        class: classString,
        url: dark,
        alt,
      }]
    },
    {
      type: container,
      class: "dark:hidden",
      children: [{
        type: "image",
        class: classString,
        url: light,
        alt,
      }]
    }
  ];
}

const directive = {
  name: 'drawio',
  doc: 'rendered drawio',
  arg: {
    type: String,
    doc: 'image path, relative to the current document',
  },
  options: {
    class: {
      type: String,
      doc: 'space-delimted classes to apply'
    },
    alt: {
      type: String,
      doc: 'alt text'
    }
  },
  run(data, vfile) {
    return render(vfile, data.arg, 'div', data.options?.class, data.options?.alt);
  },
};

const role = {
  name: 'drawio',
  doc: 'rendered drawio',
  body: {
    type: String,
    required: true,
  },
  run(data, vfile) {
    var path = data.arg;
    var alt = undefined;
    const pipe = path.indexOf('|');
    if (pipe >= 0) {
      alt = path.slice(0, pipe);
      if (!alt) {
        alt = undefined;
      }
      path = path.slice(pipe);
    }
    return render(vfile, path, 'span', undefined, alt);
  },
};

const plugin = {
  name: 'drawio',
  directives: [directive],
  roles: [role],
};

export default plugin;
