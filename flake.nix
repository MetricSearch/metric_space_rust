{
  description = "Metric Space Research Kit";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          inherit (pkgs) lib;
          # pynndescent is marked broken on aarch64-linux
          hasPynndescent = lib.meta.availableOn pkgs.stdenv.hostPlatform pkgs.python3Packages.pynndescent;
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              cargo
              clippy
              rust-analyzer
              rustc
              rustfmt

              # hdf5-metno's `static` feature builds HDF5 from source
              cmake
              # .cargo/config.toml sets `net.git-fetch-with-cli`
              git
            ];

            RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
          };
        }
        // lib.optionalAttrs hasPynndescent {
          # For the notebooks in sisap2025/python and rp_forest/sisap2026/python
          notebooks = pkgs.mkShell {
            packages = [
              (pkgs.python3.withPackages (
                ps: with ps; [
                  h5py
                  jupyterlab
                  matplotlib
                  numpy
                  pynndescent
                  scipy
                ]
              ))
            ];
          };
        }
      );

      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt);
    };
}
