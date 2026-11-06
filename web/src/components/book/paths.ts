export const BOOK = "/every-byte";

export const bookHref = (key: string) => (key === "contents" ? `${BOOK}/contents` : key === "errata" ? `${BOOK}/contents#errata` : `${BOOK}/${key}`);
