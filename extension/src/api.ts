export const VisualizerConfigurationBaseName: string = 'timeTravellingVisualizer';
export const VisualizationDataTypeConfigurationName: string = 'loadVisualization.dataType';
export const VisualizationTaskTypeConfigurationName: string = 'loadVisualization.taskType';
export const VisualizationContentPathConfigurationName: string = 'loadVisualization.contentPath';
export const VisualizationMethodConfigurationName: string = 'loadVisualization.visualizationMethod';

export class StringSelection {
    readonly selections: Set<string> = new Set<string>();
    constructor (...args: string[]) {
        args.forEach((arg) => {
            this.selections.add(arg);
        });
    }
    is(arg: any): arg is string {
        return this.selections.has(arg);
    }
}

export const VisualizationDataType: StringSelection = new StringSelection('Image', 'Text');
export const VisualizationTaskType: StringSelection = new StringSelection('Classification', 'Non-Classification');
export const VisualizationMethod: StringSelection = new StringSelection('TrustVis');