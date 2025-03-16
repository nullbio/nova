# Nova Language Support

This directory contains files needed to create syntax highlighting extensions for Nova language in VS Code, Cursor, and other editors that support TextMate grammars.

## Files

- `nova-language-configuration.json`: Contains editor configuration for Nova language (comments, brackets, auto-closing pairs, etc.)
- `nova.tmLanguage.json`: TextMate grammar for Nova language syntax highlighting

## Creating a VS Code Extension

1. Install the VS Code Extension Generator:
   ```
   npm install -g yo generator-code
   ```

2. Generate a new extension:
   ```
   yo code
   ```
   Select "New Language Support" and follow the prompts.

3. Replace the generated language configuration and grammar files with the ones from this directory.

4. Update `package.json` to include:
   ```json
   "contributes": {
     "languages": [{
       "id": "nova",
       "aliases": ["Nova", "nova"],
       "extensions": [".nova"],
       "configuration": "./language-configuration.json"
     }],
     "grammars": [{
       "language": "nova",
       "scopeName": "source.nova",
       "path": "./syntaxes/nova.tmLanguage.json"
     }]
   }
   ```

5. Add the following to support embedded Nova code in Python files:
   ```json
   "grammars": [{
     "injectTo": ["source.python"],
     "scopeName": "source.nova.embedded.python",
     "path": "./syntaxes/nova-embedded.json"
   }]
   ```

6. Create `nova-embedded.json` for embedding Nova in Python files (see example below).

## Embedding Nova in Python

To support Nova code highlighting within Python files (as comments or strings), create a `nova-embedded.json` file:

```json
{
  "scopeName": "source.nova.embedded.python",
  "injectionSelector": "L:source.python",
  "patterns": [
    {
      "include": "#nova-code-block"
    }
  ],
  "repository": {
    "nova-code-block": {
      "begin": "(# NOVA CODE START|'''nova|\"\"\"nova)",
      "end": "(# NOVA CODE END|'''|\"\"\")",
      "contentName": "source.nova",
      "patterns": [
        { "include": "source.nova" }
      ]
    }
  }
}
```

This will highlight Nova code in Python files when it's enclosed in:
- Triple quotes with "nova" identifier: `'''nova` or `"""nova`
- Special comments: `# NOVA CODE START` and `# NOVA CODE END`

## Creating a Cursor Extension

Cursor is based on VS Code, so the same extension format works with minimal modifications.

## Customizing Colors

You can customize the colors in your VS Code/Cursor settings:

```json
"editor.tokenColorCustomizations": {
  "textMateRules": [
    {
      "scope": "keyword.control.nova",
      "settings": {
        "foreground": "#5046E4"
      }
    },
    {
      "scope": "entity.name.function.nova",
      "settings": {
        "foreground": "#FF4081"
      }
    },
    {
      "scope": "constant.numeric.nova",
      "settings": {
        "foreground": "#FF8F00"
      }
    },
    {
      "scope": "comment.line.number-sign.nova",
      "settings": {
        "foreground": "#757575",
        "fontStyle": "italic"
      }
    },
    {
      "scope": "string.quoted.double.nova, string.quoted.single.nova",
      "settings": {
        "foreground": "#43A047"
      }
    }
  ]
}
```

## Publishing Your Extension

1. Package your extension:
   ```
   vsce package
   ```

2. Publish to VS Code Marketplace (requires a Microsoft account):
   ```
   vsce publish
   ```

3. For Cursor, publish the extension to VS Code Marketplace and users can install it from there.