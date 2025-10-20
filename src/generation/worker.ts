import {Request, Result} from "./message";

import init, {greet} from "../my-lib/pkg"

await init()
greet();

onmessage = (e: MessageEvent<Request>) => {

    const result: Result = {
        id: e.data.id,
        result: e.data.args[0] + e.data.args[1]
    }

    setTimeout(() => {
        postMessage(result);
    }, 1000)


};