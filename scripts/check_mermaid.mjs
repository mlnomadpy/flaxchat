/** Parse extracted Mermaid diagrams without launching a browser. */

import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [diagramDirectory, mermaidModule, jsdomModule] = process.argv.slice(2);
if (!diagramDirectory || !mermaidModule || !jsdomModule) {
  console.error(
    "usage: node scripts/check_mermaid.mjs DIAGRAM_DIR MERMAID_MODULE JSDOM_MODULE",
  );
  process.exit(2);
}

const { JSDOM } = await import(pathToFileURL(resolve(jsdomModule)).href);
const dom = new JSDOM("<!doctype html><html><body></body></html>");
globalThis.window = dom.window;
globalThis.document = dom.window.document;

const { default: mermaid } = await import(pathToFileURL(resolve(mermaidModule)).href);
mermaid.initialize({ startOnLoad: false });

const diagrams = (await readdir(diagramDirectory))
  .filter((name) => name.endsWith(".mmd"))
  .sort();

if (diagrams.length === 0) {
  console.error("no Mermaid diagrams were extracted");
  process.exit(1);
}

let failures = 0;
for (const name of diagrams) {
  const source = await readFile(resolve(diagramDirectory, name), "utf8");
  try {
    await mermaid.parse(source);
  } catch (error) {
    failures += 1;
    console.error(`${name}: ${error instanceof Error ? error.message : error}`);
  }
}

if (failures > 0) {
  process.exit(1);
}
console.log(`Validated ${diagrams.length} Mermaid diagrams`);
