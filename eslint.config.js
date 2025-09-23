import eslint from '@typescript-eslint/eslint-plugin';
import parser from '@typescript-eslint/parser';

export default [
	{
		files: ['**/*.{ts,tsx}'],
		languageOptions: {
			parser: parser,
			parserOptions: {
				ecmaVersion: 'latest',
				sourceType: 'module',
				project: './tsconfig.json',
			},
		},
		plugins: {
			'@typescript-eslint': eslint,
		},
		rules: {
			...eslint.configs.recommended.rules,
			...eslint.configs['recommended-type-checked'].rules,
		},
	},
];
