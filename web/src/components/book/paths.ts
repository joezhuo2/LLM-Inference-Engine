export const BOOK = "/every-byte";

export const bookHref = (key: string) => (key === "contents" ? `${BOOK}/contents` : key === "errata" || key === "estimates" ? `${BOOK}/contents#${key}` : `${BOOK}/${key}`);
