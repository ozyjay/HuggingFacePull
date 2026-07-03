const { packager } = require("@electron/packager");

async function main() {
  const appPaths = await packager({
    dir: process.cwd(),
    name: "HuggingFacePull",
    executableName: "huggingfacepull",
    out: "out",
    overwrite: true,
    prune: true,
    ignore: [
      /^\/\.cache($|\/)/,
      /^\/\.git($|\/)/,
      /^\/\.pytest_cache($|\/)/,
      /^\/\.venv($|\/)/,
      /(^|\/)__pycache__($|\/)/,
      /^\/cache($|\/)/,
      /^\/docs($|\/)/,
      /^\/hf-test($|\/)/,
      /^\/library($|\/)/,
      /^\/models($|\/)/,
      /^\/node_modules($|\/)/,
      /^\/out($|\/)/,
      /^\/tests($|\/)/,
      /^\/.*\.egg-info($|\/)/,
    ],
  });

  for (const appPath of appPaths) {
    console.log(appPath);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
