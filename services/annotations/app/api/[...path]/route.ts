import { env } from "cloudflare:workers";
import study from "../../../data/study.json";
import { collection } from "../../../lib/collection.mjs";
export const dynamic = "force-dynamic";
const handle = (request: Request) => collection(request,{db:env.DB,study,env});
export { handle as GET, handle as POST, handle as OPTIONS };
