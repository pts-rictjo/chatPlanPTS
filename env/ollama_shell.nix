{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    # Python 3.12 och paket
    python312
    python312Packages.pip
    python312Packages.pypdf2
    python312Packages.python-docx
    python312Packages.setuptools
    python312Packages.wheel
    python312Packages.streamlit
    python312Packages.chromadb
    python312Packages.ollama
    python312Packages.openpyxl

    # LaTeX (för att kompilera sammanfattning.tex)
    texlive.combined.scheme-medium

    # Ollama CLI
    ollama
  ];

  shellHook = ''
    echo "🔹 Nix shell redo för Python + Ollama + LaTeX"
    echo "🔹 python-docx installerat – DOCX-stöd aktiverat"
    mkdir -p ~/.ollama

    # Konfig för max VRAM-användning
    cat > ~/.ollama/config.yaml <<EOF
    gpu:
      gpu_index: 0
      memory: max
    EOF

    echo "➡ Starta Ollama: ollama serve"
    echo "➡ Dra modell:    ollama pull incept5/llama3.1-claude:70b"
  '';
}
