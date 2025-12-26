const plugin = {
  name: 'Emdashes',
  transforms: [
    {
      name: 'emdash-typography',
      doc: 'Rewrites triple-dashes to emdashes',
      stage: 'document',
      plugin: (_, utils) => (node) => {
        utils.selectAll('text', node).forEach((textNode) => {
          if (textNode.value && textNode.value.includes('---')) {
            textNode.value = textNode.value.replace(/ ?--- ?/g, '—');
          }
        });
      },
    },
  ],
};

export default plugin;
