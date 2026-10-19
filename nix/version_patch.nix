{ lib, python3Packages }:
{

  patchPackageVersions =
    filepath:
    let
      depsLines = (builtins.fromTOML (builtins.readFile filepath)).project.dependencies;
      getName = depLine: builtins.elemAt (lib.strings.split ''[>=, ~=]'' depLine) 0;
      getPkgVersion = name: lib.attrsets.attrByPath [ ''${name}'' ''version'' ] ''*'' python3Packages;
      toPatchStr = old: new: ''--replace-warn "${old}" "${new}" '';
      genPatchMap = depLine: [
        ''${depLine}''
        ''${getName depLine}''
      ];
      patches = lib.lists.forEach (lib.lists.forEach depsLines genPatchMap) (
        list: (toPatchStr (builtins.elemAt list 0) (builtins.elemAt list 1))
      );
    in
    {
      subStr = ''
        ${lib.concatStrings patches} \
      '';

    };
}
