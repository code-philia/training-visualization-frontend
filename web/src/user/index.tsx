import { ConnectionStatus } from "./connection";
import { MessageHandler } from "./message";
import { ControlPanel } from "./control";

export function VisualizationOptions() {
    const useVscode = false
    return (
        <div className="user-column">
            <ConnectionStatus useVscode = {useVscode}/>
            {useVscode ? <MessageHandler /> : <ControlPanel />}
        </div>
    );
}