import { wgsl } from './wgsl-plugin';

const isDev = process.argv.includes('--dev');

const result = await Bun.build({
  entrypoints: ['./src/index.html'],
  outdir: './dist',
  plugins: [
    wgsl({ minify: !isDev})
  ],
  target: 'browser',
  minify: !isDev,
  sourcemap: isDev ? 'inline': undefined,
  define: {
    'process.env.NODE_ENV': JSON.stringify(isDev ? 'development' : 'production')
  },
  naming: {
    chunk: '[name].[ext]'
  }
});

if (!result.success) {
  console.error("Build failed!");
  for (const message of result.logs) {
    console.error(message);
  }
}