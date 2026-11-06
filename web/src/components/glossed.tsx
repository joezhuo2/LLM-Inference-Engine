import { Children, cloneElement, Fragment, isValidElement, type ReactElement, type ReactNode } from "react";
import { splitTerms } from "@/lib/glossary";
import { Term } from "./gloss";

const skip = new Set(["a", "button", "code", "pre", "kbd", "svg", "h1", "h2", "h3", "h4", "h5", "h6", "input", "textarea", "select", "option", "label"]);

function glossString(text: string): ReactNode {
  const pieces = splitTerms(text);
  if (pieces.length === 1 && !pieces[0].term) return text;
  return pieces.map((p, i) => (p.term ? <Term key={i} k={p.term}>{p.text}</Term> : p.text));
}

export function gloss(node: ReactNode): ReactNode {
  if (typeof node === "string") return glossString(node);
  if (Array.isArray(node)) return Children.map(node, gloss);
  if (!isValidElement(node)) return node;
  const el = node as ReactElement<{ children?: ReactNode; "data-no-gloss"?: unknown }>;
  if (el.type !== Fragment && typeof el.type !== "string") return node;
  if (typeof el.type === "string" && skip.has(el.type)) return node;
  if (el.props["data-no-gloss"] !== undefined || el.props.children === undefined) return node;
  return cloneElement(el, undefined, Children.map(el.props.children, gloss));
}

export function Glossed({ children }: { children: ReactNode }) {
  return <>{gloss(children)}</>;
}
