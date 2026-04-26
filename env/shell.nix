{ pkgs ? import <nixpkgs> {} }:

let
  myPyPkgs = pkgs.python312Packages.override {
    overrides = self: super: {

detmap = super.buildPythonPackage rec {
  pname = "detmap";
  version = "0.1.5";
  format = "pyproject";

  src = super.fetchPypi {
    inherit pname version;
    hash = "sha256-+dghNvneEcDM9wSIQJ+Oq9e5Q6KgquBGHWknAE8Mbu0=";
  };

  nativeBuildInputs = with self; [ hatchling ];

  propagatedBuildInputs = with self; [
    numpy
    requests
    jax
  ];

  pythonRuntimeDepsCheck = false;
};

    };
  };

in
pkgs.mkShell {
  packages =
    (with pkgs; [
      python312
      ollama
      texlive.combined.scheme-medium
      python312Packages.pip
      python312Packages.matplotlib
      python312Packages.mplcursors
      python312Packages.ollama
      python312Packages.whoosh
    ])
    ++
    (with myPyPkgs; [
      pip
      pypdf2
      python-docx
      setuptools
      wheel
      streamlit
      chromadb
      openpyxl
      pypdf
      tabula-py
      rank-bm25
      sentence-transformers
    ]);

  shellHook = ''
    rm -rf .venv
    python -m venv .venv
    source .venv/bin/activate

    pip install --upgrade pip
    pip install "pandas>=2.3.1,<3" 
    pip install detmap
    pip install streamlit chromadb sentence-transformers

    echo "🔹 Nix shell redo för Python + Ollama + LaTeX"
    echo "🔹 python-docx installerat – DOCX-stöd aktiverat"

    mkdir -p ~/.ollama

    cat > ~/.ollama/config.yaml <<EOF
gpu:
  gpu_index: 0
  memory: max
EOF

    echo "➡ Starta Ollama: ollama serve"
    echo "➡ Dra modell:    ollama pull incept5/llama3.1-claude:70b"
  '';
}
