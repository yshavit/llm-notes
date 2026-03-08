import { parse, join, relative } from 'node:path';
import { existsSync } from 'node:fs';

const plugin = {
  name: 'in-img',
  roles: [
    {
      name: 'in-img',
      doc: 'inline image',
      body: {
        type: String,
        required: true,
      },
      run(data, vfile) {
        const { dir, name, ext } = parse(data.body);
        function filePath(theme) {
          var pathWithTheme = join(vfile.dirname, dir, `${name}-${theme}${ext}`);
          if (!existsSync(pathWithTheme)) {
            throw new Error(`missing file: ${pathWithTheme} (cwd=${vfile.cwd})`);
          }
          pathWithTheme = relative(vfile.dirname, pathWithTheme);
          return pathWithTheme;
        }

        const alt = 'TODO';

        return [
          {
            type: 'span',
            class: "hidden dark-only inline-icon",
            children: [{
              type: "image",
              url: filePath('dark'),
              alt,
            }]
          },
          {
            type: 'span',
            class: "dark:hidden inline-icon",
            children: [{
              type: "image",
              url: filePath('light'),
              alt,
            }]
          }
        ];
      },
    },
  ]
};

export default plugin;
