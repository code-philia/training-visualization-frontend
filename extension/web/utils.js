
const NUM_POINTS_FOG_THRESHOLD = 5000;
const MIN_POINT_SIZE = 5;
const IMAGE_SIZE = 30;
// Constants relating to the indices of buffer arrays.
const RGB_NUM_ELEMENTS = 3;
const INDEX_NUM_ELEMENTS = 1;
const XYZ_NUM_ELEMENTS = 3;

function cleanMaterial(material) {


    // 释放纹理
    if (material.map) material.map.dispose();
    if (material.lightMap) material.lightMap.dispose();
    if (material.bumpMap) material.bumpMap.dispose();
    if (material.normalMap) material.normalMap.dispose();
    if (material.specularMap) material.specularMap.dispose();
    if (material.envMap) material.envMap.dispose();

    material.dispose();
    // ...处理其他类型的纹理
}

function updateFixedHoverLabel(x, y, index, flag, canvas, labelType, isDisplay) {
    let specifiedFixedHoverLabel = makeSpecifiedVariableName(labelType, flag);
    console.log("specifiedHoverLabel", specifiedFixedHoverLabel);
    const label = document.getElementById(specifiedFixedHoverLabel);
    if (!isDisplay) {
        label.style.display = 'none';
        return;
    }

    var rect = canvas.getBoundingClientRect();

    // make sure selected index are not shown outside of viewport
    if (x > rect.right || y > rect.bottom || x < rect.left || y < rect.top) {
        label.style.display = 'none';
    } else {
        label.style.left = `${x + 2}px`;
        label.style.top = `${y - 2}px`;
        label.textContent = `${index}`;
        label.style.display = 'block';
    }
}

function updateCurrHoverIndex(event, index, isDisplay, flag) {
    let specifiedHoverLabel = makeSpecifiedVariableName('hoverLabel', flag);
    const hoverLabel = document.getElementById(specifiedHoverLabel);
    if (isDisplay) {
        hoverLabel.style.left = (event.clientX + 5) + 'px';
        hoverLabel.style.top = (event.clientY - 5) + 'px';
        hoverLabel.style.display = 'block';
    } else {
        if (index != null) {
            let specifiedHoverIndex = makeSpecifiedVariableName('hoverIndex', flag);
            window.vueApp[specifiedHoverIndex] = index;
            hoverLabel.textContent = `${index}`;
            hoverLabel.style.left = (event.clientX + 5) + 'px';
            hoverLabel.style.top = (event.clientY - 5) + 'px';
            hoverLabel.style.display = 'block';

        } else {
            if (hoverLabel) {
                hoverLabel.textContent = '';
                hoverLabel.style.display = 'none';
            }
        }
    }


}

function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

function makeSpecifiedVariableName(string, flag) {
    if (flag != "") {
        return string + capitalizeFirstLetter(flag);
    }
    return string;
}


function drawTimeline(res, flag) {
    console.log('res', res);
    // this.d3loader()

    const d3 = window.d3;
    let specifiedTimeLinesvg = makeSpecifiedVariableName('timeLinesvg', flag);
    let specifiedContentPath = makeSpecifiedVariableName('contentPath', flag);
    let specifiedCurrEpoch = makeSpecifiedVariableName('currEpoch', flag);

    let currEpoch = window.vueApp[specifiedCurrEpoch] ?? window.vueApp.currEpoch;

    let svgDom = document.getElementById(specifiedTimeLinesvg);


    while (svgDom?.firstChild) {
        svgDom.removeChild(svgDom.lastChild);
    }



    let total = res.structure.length;

    window.treejson = res.structure;

    let data = res.structure;


    function tranListToTreeData(arr) {
        const newArr = [];
        const map = {};

        arr.forEach(item => {
            item.children = [];
            const key = item.value;
            map[key] = item;
        });

        // 2. 对于arr中的每一项
        arr.forEach(item => {
            const parent = map[item.pid];
            if (parent) {
                //    如果它有父级，把当前对象添加父级元素的children中
                parent.children.push(item);
            } else {
                //    如果它没有父级（pid:''）,直接添加到newArr
                newArr.push(item);
            }
        });

        return newArr;
    }
    data = tranListToTreeData(data)[0];
    var margin = 20;
    var svg = d3.select(svgDom);
    var width = svg.attr("width");
    var height = svg.attr("height");

    //create group
    var g = svg.append("g")
        .attr("transform", "translate(" + margin + "," + 0 + ")");


    //create layer layout
    var hierarchyData = d3.hierarchy(data)
        .sum(function (d, i) {
            return d.value;
        });

    //create tree
    // The number of links is 1 less than number of nodes
    let len = total - 1;

    let svgWidth = len * 40;
    if (window.sessionStorage.taskType === 'active learning') {
        svgWidth = 1000;
    }
    // svgWidth = 1000
    console.log('svgWid', len, svgWidth);
    svgDom.style.width = svgWidth + 200;
    if (window.sessionStorage.selectedSetting !== 'active learning' && window.sessionStorage.selectedSetting !== 'dense al') {
        svgDom.style.height = 60;
        // svgDom.style.width = 2000
    }

    // TODO Why we need to draw the tree manually although its size is determined here?
    var tree = d3.tree()
        .size([100, svgWidth])
        .separation(function (a, b) {
            return (a.parent == b.parent ? 1 : 2) / a.depth;
        });

    //init
    var treeData = tree(hierarchyData);

    //line node
    var nodes = treeData.descendants();
    var links = treeData.links();

    //line
    var link = d3.linkHorizontal()
        .x(function (d) {
            return d.y;
        }) //linkHorizontal
        .y(function (d) {
            return d.x;
        });


    const purple = '#452d8a';
    const blue = 'rgb(26,80, 188)';
    const w_blue = 'rgb(131, 150, 188)';

    //path
    g.append('g')
        .selectAll('path')
        .data(links)
        .enter()
        .append('path')
        .attr('d', function (d, i) {
            var start = {
                x: d.source.x,
                y: d.source.y
            };
            var end = {
                x: d.target.x,
                y: d.target.y
            };
            return link({
                source: start,
                target: end
            });
        })
        .attr('stroke', w_blue)
        .attr('stroke-width', 1)
        .attr('fill', 'none');


    //创建节点与文字分组
    var gs = g.append('g')
        .selectAll('.g')
        .data(nodes)
        .enter()
        .append('g')
        .attr('transform', function (d, i) {
            return 'translate(' + d.y + ',' + d.x + ')';
        });

    //绘制文字和节点
    gs.append('circle')
        .attr('r', 8)
        .attr('fill', function (d, i) {
            // console.log("1111",d.data.value, window.iteration, d.data.value == window.iteration )
            return d.data.value == currEpoch ? blue : w_blue;
        })
        .attr('stroke-width', 1)
        .attr('stroke', function (d, i) {
            return d.data.value == currEpoch ? blue : w_blue;
        })
        .attr('class', 'hover-effect');

    gs.append('text')
        .attr('x', 0)
        .attr('y', function (d, i) {
            return -14;
        })
        .attr('dy', 0)
        .attr('text-anchor', 'middle')
        .style('fill', function (d, i) {
            return d.data.value == currEpoch ? blue : w_blue;
        })
        .text(function (d, i) {
            if (window.sessionStorage.taskType === 'active learning') {
                return `${d.data.value}|${d.data.name}`;
            } else {
                return `${d.data.value}`;
            }

        });
    setTimeout(() => {
        let list = svgDom.querySelectorAll("circle");
        for (let i = 0; i <= list.length; i++) {
            let c = list[i];
            if (c) {
                c.style.cursor = "pointer";
                c.addEventListener('click', (e) => {
                    if (e.target.nextSibling.innerHTML != window.vueApp[specifiedCurrEpoch]) {

                        let value = e.target.nextSibling.innerHTML.split("|")[0];
                        window.vueApp.isCanvasLoading = true;
                        if (flag != '') {
                            updateContraProjection(window.vueApp[specifiedContentPath], value, window.vueApp.taskType, flag);

                            if (window.vueApp.concurrentMode == "yes") {
                                let anotherFlag = referToAnotherFlag(flag);
                                let specifiedContentPathMirror = makeSpecifiedVariableName('contentPath', anotherFlag);
                                let specifiedCurrEpochMirror = makeSpecifiedVariableName('currEpoch', anotherFlag);

                                if (window.vueApp[specifiedCurrEpochMirror] != value) {
                                    updateContraProjection(window.vueApp[specifiedContentPathMirror], value, window.vueApp.taskType, anotherFlag);
                                    window.vueApp[specifiedCurrEpochMirror] = value;
                                    drawTimeline(res, anotherFlag);
                                    //todo res currently only support same epoch number from different content paths
                                }
                            }
                        } else {
                            updateProjection(window.vueApp[specifiedContentPath], value, window.vueApp.taskType);
                        }

                        window.sessionStorage.setItem('acceptIndicates', "");
                        window.sessionStorage.setItem('rejectIndicates', "");
                        window.vueApp[specifiedCurrEpoch] = value;
                        drawTimeline(res, flag);
                    }
                });

            }
        }
    }, 50);
}

function referToAnotherFlag(flag) {
    return flag == 'ref' ? 'tar' : 'ref';
}

function setIntersection(sets) {
    if (sets.length === 0) {
        return new Set();
    }

    // Create a copy of the first set to modify
    const intersection = new Set(sets[0]);

    // Iterate over each set and keep only elements that exist in all sets
    for (let i = 1; i < sets.length; i++) {
        const currentSet = sets[i];
        for (const element of intersection) {
            if (!currentSet.has(element)) {
                intersection.delete(element);
            }
        }
    }

    return intersection;
}

// check index point visibility in alphas visibility array 
function checkVisibility(array, index) {
    return array[index] == 1.0;
}

// update visibility of indices
function updateAlphas(alphas, indices, flipIndices, shouldShow) {
    indices.forEach(index => {
        alphas[index] = shouldShow && (!flipIndices || flipIndices.has(index)) ? 1.0 : 0.0;
    });
    return alphas;
}

// update indices to show on canvas
function updateShowingIndices(alphas, isShow, indices, flip_indices) {
    if (isShow) {
        return updateAlphas(alphas, indices, flip_indices, true);
    } else {
        return updateAlphas(alphas, indices, null, false);
    }
}

function cleanForEpochChange(flag) {
    var specifiedPointsMesh = makeSpecifiedVariableName("pointsMesh", flag);
    var specifiedOriginalSettings = makeSpecifiedVariableName("originalSettings", flag);
    console.log("specirfid point mesh", specifiedPointsMesh);
    if (window.vueApp[specifiedPointsMesh]) {
        console.log("pointMesh");
        if (window.vueApp[specifiedPointsMesh].geometry) {
            if (window.vueApp[specifiedPointsMesh].geometry.color) {
                window.vueApp[specifiedPointsMesh].geometry.color.dispose();
            }
            if (window.vueApp[specifiedPointsMesh].geometry.position) {
                window.vueApp[specifiedPointsMesh].geometry.position.dispose();
            }
            if (window.vueApp[specifiedPointsMesh].geometry.alpha) {
                window.vueApp[specifiedPointsMesh].geometry.alpha.dispose();
            }
            if (window.vueApp[specifiedPointsMesh].geometry.size) {
                window.vueApp[specifiedPointsMesh].geometry.size.dispose();
            }
            window.vueApp[specifiedPointsMesh].geometry.dispose();
        }
        if (window.vueApp[specifiedPointsMesh].material) {
            window.vueApp[specifiedPointsMesh].material.dispose();
        }
        window.vueApp[specifiedPointsMesh] = undefined;
    }

    if (window.vueApp[specifiedOriginalSettings]) {
        if (window.vueApp[specifiedOriginalSettings].originalSizes) {
            window.vueApp[specifiedOriginalSettings].originalSizes = undefined;
        }
        if (window.vueApp[specifiedOriginalSettings].originalColors) {
            window.vueApp[specifiedOriginalSettings].originalColors = undefined;
        }
    }

    if (flag == '') {
        if (window.vueApp.animationFrameId) {
            console.log("stopAnimation");
            cancelAnimationFrame(window.vueApp.animationFrameId);
            window.vueApp.animationFrameId = undefined;
        }
        if (window.vueApp.scene) {
            window.vueApp.scene.traverse(function (object) {
                if (object.isMesh) {
                    if (object.geometry) {
                        object.geometry.dispose();
                    }
                    if (object.material) {
                        if (object.material.isMaterial) {
                            cleanMaterial(object.material);
                        } else {
                            // 对于多材质的情况（材质数组）
                            for (const material of object.material) {
                                cleanMaterial(material);
                            }
                        }
                    }
                }
            });

            while (window.vueApp.scene.children.length > 0) {
                window.vueApp.scene.remove(window.vueApp.scene.children[0]);
            }
        }
        // remove previous scene
        if (window.vueApp.renderer) {
            if (container.contains(window.vueApp.renderer.domElement)) {
                console.log("removeDom");
                container.removeChild(window.vueApp.renderer.domElement);
            }
            window.vueApp.renderer.renderLists.dispose();
            window.vueApp.renderer.dispose();
        }
    } else {
        if (window.vueApp.animationFrameId[flag]) {
            console.log("stopAnimation");
            cancelAnimationFrame(window.vueApp.animationFrameId[flag]);
            window.vueApp.animationFrameId[flag] = undefined;
        }
        if (window.vueApp.scene[flag]) {
            window.vueApp.scene[flag].traverse(function (object) {
                if (object.isMesh) {
                    if (object.geometry) {
                        object.geometry.dispose();
                    }
                    if (object.material) {
                        if (object.material.isMaterial) {
                            cleanMaterial(object.material);
                        } else {
                            // 对于多材质的情况（材质数组）
                            for (const material of object.material) {
                                cleanMaterial(material);
                            }
                        }
                    }
                }
            });

            while (window.vueApp.scene[flag].children.length > 0) {
                window.vueApp.scene[flag].remove(window.vueApp.scene[flag].children[0]);
            }
        }
        // remove previous scene
        if (window.vueApp.renderer[flag]) {
            if (container.contains(window.vueApp.renderer[flag].domElement)) {
                console.log("removeDom");
                container.removeChild(window.vueApp.renderer[flag].domElement);
            }
            window.vueApp.renderer[flag].renderLists.dispose();
            window.vueApp.renderer[flag].dispose();
        }
    }

}

function resetHighlightAttributes() {
    window.vueApp.highlightAttributesRef.highlightedPointsYellow = [];
    window.vueApp.highlightAttributesRef.highlightedPointsBlue = [];
    window.vueApp.highlightAttributesRef.highlightedPointsGreen = [];
    window.vueApp.highlightAttributesTar.highlightedPointsYellow = [];
    window.vueApp.highlightAttributesTar.highlightedPointsBlue = [];
    window.vueApp.highlightAttributesTar.highlightedPointsGreen = [];
    if (window.vueApp.highlightAttributesRef.allHighlightedSet) {
        window.vueApp.highlightAttributesRef.allHighlightedSet.clear();
    }
    if (window.vueApp.highlightAttributesTar.allHighlightedSet) {
        window.vueApp.highlightAttributesTar.allHighlightedSet.clear();
    }
    window.vueApp.highlightAttributesRef.allHighlightedSet = null;
    window.vueApp.highlightAttributesTar.allHighlightedSet = null;

    window.vueApp.highlightAttributesRef.boldIndices = [];
    window.vueApp.highlightAttributesTar.boldIndices = [];
}