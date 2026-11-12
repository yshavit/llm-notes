const dfnRole = {
  name: "dfn",
  doc: "Introduces a new word to be defined.",
  body: {
    type: String,
    required: true,
    doc: "The word",
  },
  run(data) {
    return [
      { type: "html", value: "<b>" },
      { type: "text", value: data.body },
      { type: "html", value: "</b>" },
    ];
  },
};

export default {
  name: "Definition role",
  roles: [dfnRole],
};
