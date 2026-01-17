const statusDescriptions = [
  {
    title: "Stub",
    description: "This page is just a placeholder for future content.",
  },
  {
    title: "Draft",
    description: "Basics are in place, but needs review.",
  },
  {
    title: "Pretty much ready",
    description: "I've gone over this page a couple times, and I think it's in pretty good shape.",
  },
  {
    title: "Final draft",
    description: "Page should be considered complete. Further changes will be considered enhancements, not fixes.",
  }
];

const plugin = {
  name: 'Status',
  directives: [
    {
      name: 'status',
      doc: 'No-op rendering',
      arg: {
        type: String,
        doc: 'The status value, 1 - 5',
      },
      run: (data) => {
        const status = data.arg;
        if (/^\d+$/.test(status)) {
          return [{ type: 'status', status: parseInt(status) }];
        } else {
          console.error('invalid arg to status', status);
          return [];
        }
      }
    }
  ],
  transforms: [
    {
      name: 'status',
      doc: "Sets the page's status",
      stage: 'document',
      plugin: (_, _utils) => (node) => {

        const statuses = [];

        function extractAndRemoveStatuses(node) {
          if (node.children && Array.isArray(node.children)) {
            node.children = node.children.filter(child => {
              if (child.type === 'status') {
                statuses.push(child.status);
                return false; // Remove from parent
              }

              // For non-status nodes, recursively walk their children
              extractAndRemoveStatuses(child);
              return true;
            });
          }
        }
        extractAndRemoveStatuses(node);

        const icon = '⚠️';
        if (statuses.length == 0) {
          console.warn(icon, 'status not set on page');
          return;
        }
        if (statuses.length == 2) {
          console.error(icon, 'multiple statuses set on page');
          return;
        }
        const status = statuses[0];
        if (status < 0 || status >= statusDescriptions.length) {
          console.error(icon, `status must be between 0 and ${statusDescriptions.length - 1}, inclusive`);
          return;
        }
        if (!node.children) {
          console.error(icon, 'root node does not have children');
          return;
        }

        const stars = [];
        for (let i = 0; i < statusDescriptions.length - 1; i++) {
          const star = i < status ? '★' : '☆';
          stars.push(star);
        }

        const statusNode = {
          type: 'admonition',
          kind: 'attention',
          icon: false,
          class: 'simple dropdown wip-status',
          children: [
            {
              type: "admonitionTitle",
              children: [
                {
                  type: "text",
                  value: `Page Status: ${statusDescriptions[status].title} ${stars.join('')}`,
                },
              ]
            },
            {
              type: "div",
              children: [
                {
                  type: "text",
                  value: statusDescriptions[status].description,
                },
              ]
            },
          ]
        };

        node.children.unshift(statusNode);
      },
    },
  ],
};

export default plugin;
