import { useFrame, useThree } from '@react-three/fiber';
import { ContextOnlyProps, VisualizerRenderContext } from './visualizer-render-context';
import { MapControls } from '@react-three/drei';
import { OrthographicCamera, Vector3, EventListener, WebGLRenderer, Vector4 } from 'three';
import { MapControls as MapControlsImpl } from 'three-stdlib';
import { useEffect, useRef } from 'react';

type ControlState = {
    target: Vector3;
    position: Vector3;
    zoom: number;
}

// for OrbitControl.d.ts does not define the events it could dispatch
type MapControlsEventMap = {
    change: void;     // this could be incorrect
    start: void;
    end: void;
};
type MapControlsEventDispatcher = { 
    addEventListener: <T extends Extract<keyof MapControlsEventMap, string>>(
        type: T,
        listener: EventListener<MapControlsEventMap[T], T, MapControlsImpl>
    ) => void;
    removeEventListener: <T extends Extract<keyof MapControlsEventMap, string>>(
        type: T,
        listener: EventListener<MapControlsEventMap[T], T, MapControlsImpl>
    ) => void;
}

// save state externally, modified from OrbitControl definition
function saveState(mapControls: MapControlsImpl, controlState: React.MutableRefObject<ControlState | null>) {
    controlState.current = {
        target: mapControls.target,
        position: mapControls.object.position.clone(),
        zoom: mapControls.object.zoom,
    };
}

function restoreState(mapControls: MapControlsImpl, controlState: React.MutableRefObject<ControlState | null>) {
    if (mapControls && controlState.current) {
        const { target, position, zoom } = controlState.current;
        mapControls.target.copy(target);
        mapControls.object.position.copy(position);
        mapControls.object.zoom = zoom;

        mapControls.object.updateMatrixWorld();
        mapControls.update();

        // mapControls.state = MapControlsImpl.STATE.NONE;
    }
}

function resizeCamera(camera: OrthographicCamera, gl: WebGLRenderer, oldScalingFactor: number, computeOnly: boolean = false): number {
    const v4 = gl.getViewport(new Vector4());
    
    if (!computeOnly) {
        const w = v4.width / oldScalingFactor;
        const h = v4.height / oldScalingFactor;
        camera.left = -w / 2;
        camera.right = w / 2;
        camera.top = h / 2;
        camera.bottom = -h / 2;
        camera.updateProjectionMatrix();
    }
    
    // assume the viewport and camera has same aspect ratio
    // calculate factor in "pixels per unit"
    const scalingFactor = v4.width / (camera.right - camera.left);
    return scalingFactor;
}

function resetCamera(camera: OrthographicCamera, rc: VisualizerRenderContext) {
    const w = rc.initWorldWidth;
    const h = rc.initWorldHeight;
    const centerX = rc.centerX;
    const centerY = rc.centerY;

    camera.left = -w / 2;
    camera.right = w / 2;
    camera.top = h / 2;
    camera.bottom = -h / 2;
    camera.updateProjectionMatrix();
    
    camera.position.set(centerX, centerY, 100);
    camera.lookAt(centerX, centerY, 0);
    camera.up.set(0, 0, -1);     // make map control z-axis up
}

function addResizeObserver(camera: OrthographicCamera, renderer: WebGLRenderer) {
    let oldScalingFactor = resizeCamera(camera, renderer, 1, true);

    const observer = new ResizeObserver(() => {
        oldScalingFactor = resizeCamera(camera, renderer, oldScalingFactor, false);
    });

    const divElement = renderer.domElement;
    if (divElement) {
        observer.observe(divElement);
    }
}

export function VisualierDefaultControl({ visualizerRenderContext, onResize }: ContextOnlyProps & { onResize: () => void }) {
    const rc = visualizerRenderContext;
    
    const mapControlsRef = useRef<MapControlsImpl>(null);
    const controlState = useRef<ControlState | null>(null);
    
    const gl = useThree((state) => state.gl);
    const camera = useThree((state) => state.camera);

    useEffect(
        () => {
            if (mapControlsRef.current !== null) {
                rc.setControls(mapControlsRef.current);
            }
        }
    , [rc]);

    useEffect(() => {
        if (!(camera instanceof OrthographicCamera)) return;

        addResizeObserver(camera, gl);
        
        resetCamera(camera, rc);

        const mapControl = mapControlsRef.current;
        if (mapControl) {
            restoreState(mapControl, controlState);

            // let isMoving = false;
            // const listener = () => {
            //     saveState(mapControl, controlState);
            //     setResizeTrigger(new Date().toISOString());
            // };

            // (mapControl as MapControlsEventDispatcher).addEventListener('change', listener);
            // rc.canvasElement.addEventListener('mouseover', () => { if (isMoving) listener(); });
            // (mapControl as MapControlsEventDispatcher).addEventListener('start', () => { isMoving = true; });
            // (mapControl as MapControlsEventDispatcher).addEventListener('end', () => { isMoving = false; });

            // return () => {
            //     (mapControl as MapControlsEventDispatcher).removeEventListener('change', listener);
            // }
        }
    }, [gl, camera, rc, onResize]);
    
    useFrame(({ gl, scene, camera }) => {
        onResize();
        gl.render(scene, camera);
    }, 1);

    return (
        <MapControls ref={mapControlsRef}
            screenSpacePanning={true}
            enableDamping={false}
            minPolarAngle={Math.PI}
            target={[rc.centerX, rc.centerY - 0.01, 0]}
            zoomToCursor={true}
            reverseHorizontalOrbit={true}
        />
    )
}