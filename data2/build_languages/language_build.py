from tree_sitter import Language, Parser

Language.build_library(
  # Store the library in the `build` directory
  'my-languages.so',

  # Include one or more languages
  [
    # '../languages/tree-sitter-c',
    '/data/bhtian2/win_linux_mapping/three-fusion/data2/tree-sitter-cpp',
    # '../languages/tree-sitter-java'
  ]
)

# C_LANGUAGE = Language('build/my-languages.so', 'c')
# CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')