import { resolve, dirname } from "path";

/**
 * This function reads a WGSL file and recursively resolves all #import statements.
 */
async function resolveWgslImports(filePath: string, visited = new Set<string>()): Promise<string> {
  const absolutePath = resolve(filePath);
  if (visited.has(absolutePath)) return ''; // Prevents circular imports
  visited.add(absolutePath);

  if (!(await Bun.file(absolutePath).exists())) {
    throw new Error(`[WGSL Plugin] File not found: ${absolutePath}`);
  }

  const fileContent = await Bun.file(absolutePath).text();
  const fileDir = dirname(absolutePath);
  const importRegex = /#import "(.+)"/g;
  
  let finalContent = fileContent;
  for (const match of fileContent.matchAll(importRegex)) {
    const importPath = match[1];
    const includedFilePath = resolve(fileDir, importPath);
    const importedContent = await resolveWgslImports(includedFilePath, visited);
    // Replace the #import line with the resolved content
    finalContent = finalContent.replace(match[0], importedContent);
  }

  return finalContent;
}

/**
 * Minifies a WGSL (WebGPU Shading Language) string by removing comments,
 * collapsing whitespace, and removing unnecessary spaces around operators
 * and punctuation.
 *
 * @param wgslCode The WGSL code to minify.
 * @returns The minified WGSL code.
 */
export function minifyWgsl(wgslCode: string): string {
  // 1. Remove all single-line comments (//...)
  let minifiedCode = wgslCode.replace(/\/\/.*/g, '');

  // 2. Remove all multi-line comments (/*...*/) - WGSL doesn't officially have them,
  // but this makes the minifier more robust.
  minifiedCode = minifiedCode.replace(/\/\*[\s\S]*?\*\//g, '');

  // 3. Collapse all sequences of whitespace (spaces, tabs, newlines) into a single space
  minifiedCode = minifiedCode.replace(/\s+/g, ' ');

  // 4. Remove spaces around operators and punctuation, being careful not to merge tokens.
  // This regex looks for a space followed by or preceded by one of the specified characters.
  minifiedCode = minifiedCode.replace(/\s*([,;<>(){}\[\]])\s*/g, '$1');

  // 5. Remove spaces around binary and unary operators where safe.
  // We are careful with operators that can be part of other tokens (e.g., <, >).
  // A simple global replacement is generally safe for these specific operators.
  minifiedCode = minifiedCode.replace(/\s*([+\-*/%&|!^~=])\s*/g, '$1');

  // 6. Handle the specific case of `->` (return type arrow) to avoid ` - > `
  minifiedCode = minifiedCode.replace(/-\s+>/g, '->');
  
  // 7. Remove leading and trailing whitespace from the entire string
  minifiedCode = minifiedCode.trim();

  return minifiedCode;
}

export const wgsl = (options: { minify?: boolean } = {}) => ({
  name: "WGSL Importer",
  async setup(build) {
    
    build.onLoad({ filter: /\.wgsl$/ }, async (args) => {

      const source = await resolveWgslImports(args.path);
      const finalCode = options.minify ? minifyWgsl(source) : source;
      
      return {
        contents: finalCode,
        loader: "text",
      };
    });
  },
});