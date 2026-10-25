import { env } from "cloudflare:workers";
import { getChatGPTUser, chatGPTSignInPath } from "../chatgpt-auth";
import Dashboard from "./dashboard";
export const dynamic = "force-dynamic";
export default async function Admin() {
  const user = await getChatGPTUser();
  if (!user) return <main><p className="eyebrow">SOTU annotation collection</p><h1>Manage the study.</h1><p>Sign in with the ChatGPT account that owns this collection to create reviewer links and export labels.</p><a className="button" href={chatGPTSignInPath("/admin")} target="_top">Sign in with ChatGPT</a></main>;
  const owner = (env as any).OWNER_EMAIL;
  if (!owner || user.email.toLowerCase() !== owner.toLowerCase()) return <main><h1>Owner access required.</h1><p>This dashboard is available only to the study owner.</p></main>;
  return <Dashboard />;
}
