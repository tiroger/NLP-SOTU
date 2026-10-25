import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {title:"SOTU Review Collection",description:"Private storage and assignment management for independent State of the Union sentiment annotation.",robots:{index:false,follow:false}};
export default function RootLayout({children}:Readonly<{children:React.ReactNode}>) { return <html lang="en"><body>{children}</body></html>; }
