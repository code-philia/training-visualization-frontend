import * as vscode from 'vscode';
import * as config from './config';
import { MetadataViewManager } from './views/metadataView';

export function doViewsRegistration(): vscode.Disposable {
    const metadataViewRegistration = vscode.window.registerWebviewViewProvider(
        config.ViewsID.metadataView,
        MetadataViewManager.getWebViewProvider(),
        { webviewOptions: { retainContextWhenHidden: true } }
    );
    return vscode.Disposable.from(metadataViewRegistration);
}