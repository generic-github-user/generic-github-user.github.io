let
  pkgs = import <nixpkgs> {};
  python = pkgs.python311;
  pythonPackages = python.pkgs;
in

with pkgs;

mkShell {
  name = "pip-env";
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.zlib
    pkgs.libGL
    pkgs.glib
    pkgs.stdenv.cc.cc
    pkgs.blas
    pkgs.lapack
    pkgs.gsl
  ];
  buildInputs = with pythonPackages; [
    zlib
    pkgs.uv
    pkgs.less

    pkgs.sqlite
  ];
  shellHook = ''
    alias gst="git status"
    alias gc="git commit"
    alias gca="git commit -a"
    alias gp="git push"
    alias ga="git add"
    alias gd="git diff"
  '';
}

