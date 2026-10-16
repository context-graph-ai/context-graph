//! Parse store APIs in process and follow local definitions to named-table SQL.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};

struct Definition {
    name: String,
    path: String,
    public_method: bool,
    words: Vec<String>,
    strings: Vec<(String, usize)>,
}

#[derive(Default)]
struct References {
    words: Vec<String>,
    strings: Vec<(String, usize)>,
}

fn test_only(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| match &attr.meta {
        syn::Meta::List(meta) => meta.path.is_ident("cfg") && meta.tokens.to_string() == "test",
        _ => false,
    })
}

impl<'ast> Visit<'ast> for References {
    fn visit_ident(&mut self, ident: &'ast syn::Ident) {
        self.words.push(ident.to_string());
    }

    fn visit_lit_str(&mut self, literal: &'ast syn::LitStr) {
        self.strings
            .push((literal.value(), literal.span().start().line));
    }

    fn visit_lit_byte_str(&mut self, literal: &'ast syn::LitByteStr) {
        self.strings.push((
            String::from_utf8_lossy(&literal.value()).into_owned(),
            literal.span().start().line,
        ));
    }

    fn visit_item(&mut self, item: &'ast syn::Item) {
        if !item_test_only(item) {
            visit::visit_item(self, item);
        }
    }

    fn visit_macro(&mut self, item: &'ast syn::Macro) {
        // Macro bodies are token streams; preserve the lexical rule's conservative references.
        self.token_stream(item.tokens.clone());
    }
}

impl References {
    fn token_stream(&mut self, stream: proc_macro2::TokenStream) {
        for token in stream {
            match token {
                proc_macro2::TokenTree::Group(group) => self.token_stream(group.stream()),
                proc_macro2::TokenTree::Ident(ident) => self.words.push(ident.to_string()),
                proc_macro2::TokenTree::Literal(literal) => {
                    if let Ok(value) = syn::parse_str::<syn::LitStr>(&literal.to_string()) {
                        self.strings
                            .push((value.value(), literal.span().start().line));
                    } else if let Ok(value) =
                        syn::parse_str::<syn::LitByteStr>(&literal.to_string())
                    {
                        self.strings.push((
                            String::from_utf8_lossy(&value.value()).into_owned(),
                            literal.span().start().line,
                        ));
                    }
                }
                _ => {}
            }
        }
    }
}

fn item_test_only(item: &syn::Item) -> bool {
    let attrs = match item {
        syn::Item::Fn(item) => &item.attrs,
        syn::Item::Const(item) => &item.attrs,
        syn::Item::Static(item) => &item.attrs,
        syn::Item::Mod(item) => &item.attrs,
        syn::Item::Impl(item) => &item.attrs,
        _ => return false,
    };
    test_only(attrs)
}

fn definitions(items: &[syn::Item], path: &str, found: &mut Vec<Definition>) {
    for item in items {
        if item_test_only(item) {
            continue;
        }
        let mut references = References::default();
        let name = match item {
            syn::Item::Fn(item) => {
                references.visit_block(&item.block);
                nested_definitions(&item.block, path, found);
                item.sig.ident.to_string()
            }
            syn::Item::Const(item) => {
                references.visit_expr(&item.expr);
                item.ident.to_string()
            }
            syn::Item::Static(item) => {
                references.visit_expr(&item.expr);
                item.ident.to_string()
            }
            syn::Item::Mod(item) => {
                if let Some((_, items)) = &item.content {
                    definitions(items, path, found);
                }
                continue;
            }
            syn::Item::Impl(item) => {
                let store_view = item.trait_.is_none()
                    && matches!(item.self_ty.as_ref(),
                    syn::Type::Path(ty) if ty.path.segments.last().is_some_and(|segment|
                        matches!(segment.ident.to_string().as_str(), "Store" | "ScopedStore" | "ConsumerGraphView")));
                for member in &item.items {
                    let mut references = References::default();
                    let (name, public_method) = match member {
                        syn::ImplItem::Fn(method) if !test_only(&method.attrs) => {
                            references.visit_block(&method.block);
                            nested_definitions(&method.block, path, found);
                            (
                                method.sig.ident.to_string(),
                                store_view && matches!(method.vis, syn::Visibility::Public(_)),
                            )
                        }
                        syn::ImplItem::Const(constant) if !test_only(&constant.attrs) => {
                            references.visit_expr(&constant.expr);
                            (constant.ident.to_string(), false)
                        }
                        _ => continue,
                    };
                    found.push(Definition {
                        name,
                        path: path.into(),
                        public_method,
                        words: references.words,
                        strings: references.strings,
                    });
                }
                continue;
            }
            syn::Item::Trait(item) => {
                for member in &item.items {
                    let mut references = References::default();
                    let name = match member {
                        syn::TraitItem::Fn(method) if !test_only(&method.attrs) => {
                            let Some(block) = &method.default else {
                                continue;
                            };
                            references.visit_block(block);
                            nested_definitions(block, path, found);
                            method.sig.ident.to_string()
                        }
                        syn::TraitItem::Const(constant) if !test_only(&constant.attrs) => {
                            let Some((_, expression)) = &constant.default else {
                                continue;
                            };
                            references.visit_expr(expression);
                            constant.ident.to_string()
                        }
                        _ => continue,
                    };
                    found.push(Definition {
                        name,
                        path: path.into(),
                        public_method: false,
                        words: references.words,
                        strings: references.strings,
                    });
                }
                continue;
            }
            _ => continue,
        };
        found.push(Definition {
            name,
            path: path.into(),
            public_method: false,
            words: references.words,
            strings: references.strings,
        });
    }
}

fn nested_definitions(block: &syn::Block, path: &str, found: &mut Vec<Definition>) {
    struct Nested<'a> {
        path: &'a str,
        found: &'a mut Vec<Definition>,
    }
    impl<'ast> Visit<'ast> for Nested<'_> {
        fn visit_item(&mut self, item: &'ast syn::Item) {
            definitions(std::slice::from_ref(item), self.path, self.found);
        }
    }
    Nested { path, found }.visit_block(block);
}

fn source_files(path: &Path, files: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(path).expect("source directory") {
        let path = entry.expect("source entry").path();
        if path.is_dir() {
            source_files(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
}

fn named_tables(sql: &str) -> Vec<String> {
    // A SQL keyword, whitespace, optional identifier quote, and an exact table word.
    let mut tables = Vec::new();
    let bytes = sql.as_bytes();
    let word = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    let mut i = 0;
    while i < bytes.len() {
        if !word(bytes[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && word(bytes[i]) {
            i += 1;
        }
        if !matches!(
            sql[start..i].to_ascii_uppercase().as_str(),
            "FROM" | "JOIN" | "INTO" | "UPDATE" | "USING"
        ) {
            continue;
        }
        let gap = i;
        // Match Unicode whitespace and the additional file/group/record separators recognized by
        // Python's whitespace rules.
        while let Some(ch) = sql[i..].chars().next() {
            if !ch.is_whitespace() && !matches!(ch, '\u{1c}'..='\u{1f}') {
                break;
            }
            i += ch.len_utf8();
        }
        if i == gap {
            continue;
        }
        if i < bytes.len() && matches!(bytes[i], b'"' | b'`' | b'[') {
            i += 1;
        }
        let start = i;
        while i < bytes.len() && word(bytes[i]) {
            i += 1;
        }
        let table = sql[start..i].to_ascii_lowercase();
        if matches!(table.as_str(), "signatures" | "blueprints" | "approvals") {
            tables.push(table);
        }
    }
    tables
}

pub fn scan(root: &Path) -> Vec<serde_json::Value> {
    let mut files = Vec::new();
    source_files(&root.join("crates/context-graph/src"), &mut files);
    files.sort();
    let mut found = Vec::new();
    for file in files {
        let source = std::fs::read_to_string(&file).expect("Rust source");
        let parsed =
            syn::parse_file(&source).unwrap_or_else(|error| panic!("{}: {error}", file.display()));
        definitions(
            &parsed.items,
            &file.strip_prefix(root).unwrap().to_string_lossy(),
            &mut found,
        );
    }
    let mut by_name: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, item) in found.iter().enumerate() {
        by_name.entry(&item.name).or_default().push(index);
    }
    let mut violations = Vec::new();
    for (start, entry) in found
        .iter()
        .enumerate()
        .filter(|(_, item)| item.public_method)
    {
        let mut todo = vec![(start, vec![entry.name.clone()])];
        let mut seen = BTreeSet::new();
        while let Some((index, chain)) = todo.pop() {
            if !seen.insert(index) {
                continue;
            }
            let item = &found[index];
            for (sql, line) in &item.strings {
                for table in named_tables(sql) {
                    violations.push(
                        serde_json::json!({"api": entry.name, "api_path": entry.path,
                        "path": item.path, "line": line, "table": table, "chain": chain}),
                    );
                }
            }
            for word in &item.words {
                for &callee in by_name.get(word.as_str()).into_iter().flatten() {
                    if found[callee].path == item.path || found[callee].path.contains("/src/store/")
                    {
                        let mut next_chain = chain.clone();
                        next_chain.push(word.clone());
                        todo.push((callee, next_chain));
                    }
                }
            }
        }
    }
    violations.sort_by_key(|hit| {
        (
            hit["api_path"].as_str().unwrap().to_string(),
            hit["api"].as_str().unwrap().to_string(),
            hit["path"].as_str().unwrap().to_string(),
            hit["line"].as_u64().unwrap(),
            hit["table"].as_str().unwrap().to_string(),
        )
    });
    violations
}
