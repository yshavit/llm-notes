
const plugin = {
  name: 'emoticons',
  roles: [
    {
      name: 'emoticon',
      doc: 'inline text as emoticons',
      body: {
        type: String,
        required: true,
      },
      run(data) {
        return [{
          type: 'span',
          class: 'emoticon gray-600 dark:text-gray-400',
          children: [{
            type: 'text',
            value: data.node.value,
          }],
        }];
      },
    },
  ]
};

export default plugin;
