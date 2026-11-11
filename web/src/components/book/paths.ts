export const BOOK = "/every-byte";

export const bookHref = (key: string) => (key === "contents" ? `${BOOK}/contents` : key === "errata" || key === "estimates" || key === "beyond" ? `${BOOK}/contents#${key}` : `${BOOK}/${key}`);
