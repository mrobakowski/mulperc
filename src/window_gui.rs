extern crate rand;

use conrod;
use std;
use nfd;
use multilayer_perceptron::{MultilayerPerceptron, NetFile};
use std::collections::HashMap;
use std::path::Path;
use bincode;
use na::Iterable;
use mnist::MnistDigits;
use std::ops::{Deref, DerefMut};

pub const WIN_W: u32 = 800;
pub const WIN_H: u32 = 720;

pub struct AppState {
    pub classifier: ClassifierState,
    pub autoencoder: AutoencoderState,
    pub mnist: MnistPreviewState
}

pub struct WithPath<T> {
    data: Option<T>,
    path: Option<String>
}

impl<T> WithPath<T> {
    fn new() -> Self {
        WithPath {
            data: None,
            path: None,
        }
    }
}

impl<T> Deref for WithPath<T> {
    type Target = Option<T>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for WithPath<T> {
    fn deref_mut(&mut self) -> &mut Option<T> {
        &mut self.data
    }
}

pub struct ClassifierState {
    pub image: WithPath<Vec<f64>>,
    pub net: WithPath<NetFile>
}

pub struct AutoencoderState {
    pub net: WithPath<MultilayerPerceptron>,
    pub src_image: Option<String>
}

pub struct MnistPreviewState {
    pub idx: usize
}

impl AppState {
    pub fn new() -> Self {
        AppState {
            classifier: ClassifierState {
                image: WithPath::new(),
                net: WithPath::new(),
            },
            autoencoder: AutoencoderState {
                net: WithPath::new(),
                src_image: None,
            },
            mnist: MnistPreviewState {
                idx: 0
            },
        }
    }
}

pub fn theme() -> conrod::Theme {
    conrod::Theme {
        name: "Demo Theme".to_string(),
        padding: conrod::Padding::none(),
        x_position: conrod::Position::Align(conrod::Align::Start, None),
        y_position: conrod::Position::Direction(conrod::Direction::Backwards, 20.0, None),
        background_color: conrod::color::DARK_CHARCOAL,
        shape_color: conrod::color::LIGHT_CHARCOAL,
        border_color: conrod::color::BLACK,
        border_width: 0.0,
        label_color: conrod::color::WHITE,
        font_id: None,
        font_size_large: 26,
        font_size_medium: 18,
        font_size_small: 12,
        widget_styling: std::collections::HashMap::new(),
        mouse_drag_threshold: 0.0,
        double_click_threshold: std::time::Duration::from_millis(500),
    }
}

pub fn image_map<T>(
    ids: &Ids,
) -> conrod::image::Map<T> {
    image_map! {

    }
}


widget_ids! {
    pub struct Ids {
        main_canvas,
        tabs,
        tabs_scrollbar,

        classifier_tab,
        mnist_preview_tab,
        autoencoder_tab,

        classifier_canvas,
        mnist_preview_canvas,
        autoencoder_canvas,

        classifier_scrollbar,
        mnist_preview_scrollbar,
        autoencoder_scrollbar,
    }
}


pub fn gui(ui: &mut conrod::UiCell, ids: &Ids, app: &mut AppState) {
    use conrod::{widget, Labelable, Positionable, Sizeable, Widget};

    const MARGIN: conrod::Scalar = 30.0;
    const GAP: conrod::Scalar = 20.0;

    widget::Canvas::new().pad(MARGIN).set(ids.main_canvas, ui);

    widget::Tabs::new(
        &[
            (ids.classifier_tab, "Classifier"),
            (ids.mnist_preview_tab, "MNIST Preview"),
            (ids.autoencoder_tab, "Autoencoder")
        ])
        .canvas_style({
            let mut s = conrod::widget::canvas::Style::new();
            s.color = Some(conrod::color::DARK_ORANGE);
            s
        })
        .bar_thickness(80f64)
        .wh_of(ids.main_canvas)
        .middle_of(ids.main_canvas)
        .set(ids.tabs, ui);

    classifier_tab(ui, ids, &mut app.classifier);
    mnist_preview_tab(ui, ids, &mut app.mnist);
    autoencoder_tab(ui, ids, &mut app.autoencoder);
    {
        //
        //    widget::Text::new(&app.net_path)
        //        .align_text_middle()
        //        .mid_top_of(ids.classifier_canvas)
        //        .set(ids.title, ui);
        //
        //    for _ in widget::Button::new()
        //        .label("Image")
        //        .top_left_of(ids.classifier_canvas)
        //        .down(MARGIN)
        //        .w_h(130.0, 70.0)
        //        .set(ids.image_path_btn, ui)
        //        {
        //            if let Ok(response) = nfd::open_file_dialog(None, None) {
        //                if let nfd::Response::Okay(path) = response {
        //                    app.image_path = path;
        //                }
        //            }
        //        }
        //
        //    fn open_net<P: AsRef<Path>>(p: P) -> Option<(MultilayerPerceptron, HashMap<usize, String>)> {
        //        use std::fs::File;
        //        struct IDontCare;
        //        impl<T: std::error::Error> From<T> for IDontCare {
        //            fn from(_: T) -> Self {
        //                IDontCare
        //            }
        //        }
        //        let res: Result<_, IDontCare> = (|| {
        //            let mut file = File::open(p)?;
        //            let NetFile(perc, ntl) = bincode::serde::deserialize_from(&mut file, bincode::SizeLimit::Infinite)?;
        //
        //            Ok((perc, ntl))
        //        })();
        //        res.ok()
        //    }
        //
        //    for _ in widget::Button::new()
        //        .label("Net")
        //        .mid_right_of(ids.classifier_canvas)
        //        .align_middle_y_of(ids.image_path_btn)
        //        .set(ids.net_path_btn, ui)
        //        {
        //            if let Ok(response) = nfd::open_file_dialog(None, None) {
        //                if let nfd::Response::Okay(path) = response {
        //                    if let x @ Some(..) = open_net(&path) {
        //                        app.net_path = path;
        //                        app.net = x;
        //                    }
        //                }
        //            }
        //        }
        //
        //
        //    let mult = 25.0;
        //
        //    widget::Image::new()
        //        .w_h(7.0 * mult, 10.0 * mult)
        //        .mid_left_of(ids.classifier_canvas)
        //        .down(GAP)
        //        .set(ids.preview_image, ui);
        //
        //    if let Some((ref perc, ref labels)) = app.net {
        //        if let Some(ref image) = app.image {
        //            let out = perc.feed_forward(&*image).0;
        //            let decoded = &labels[&out.iter().enumerate()
        //                .max_by(|a, b|
        //                    a.1.partial_cmp(b.1).unwrap()
        //                ).unwrap().0];
        //
        //            widget::Text::new(&decoded)
        //                .align_text_middle()
        //                .mid_right_with_margin_on(ids.classifier_canvas, MARGIN)
        //                .align_middle_y_of(ids.preview_image)
        //                .font_size(150)
        //                .set(ids.label, ui);
        //        }
        //    }
        //
        //    for val in widget::NumberDialer::new(app.mnist_idx as f64, 0f64, 59999f64, 0)
        //        .w_h(130.0, 70.0)
        //        .down(GAP)
        //        .align_middle_x_of(ids.classifier_canvas)
        //        .set(ids.mnist_idx, ui)
        //        {
        //            app.mnist_idx = val as usize;
        //        }
        //
        //    widget::Image::new()
        //        .w_h(280.0, 280.0)
        //        .down(GAP)
        //        .align_middle_x_of(ids.classifier_canvas)
        //        .set(ids.mnist_img, ui);
        //
        //    widget::Text::new(&app.mnist[app.mnist_idx].1)
        //        .down(GAP)
        //        .align_middle_x_of(ids.classifier_canvas)
        //        .set(ids.mnist_label, ui);
        //
        //
        //    ///////// AUTOENCODER //////////
        //
        //    widget::Text::new(&app.autoencoder_path)
        //        .align_text_middle()
        //        .mid_top_of(ids.autoencoder)
        //        .set(ids.title_autoencoder, ui);
        //
        //    for _ in widget::Button::new()
        //        .label("Image")
        //        .top_left_of(ids.autoencoder)
        //        .down(MARGIN)
        //        .w_h(130.0, 70.0)
        //        .set(ids.image_path_btn_autoencoder, ui)
        //        {
        //            if let Ok(response) = nfd::open_file_dialog(None, None) {
        //                if let nfd::Response::Okay(path) = response {
        //                    app.image_path_autoencoder = path;
        //                }
        //            }
        //        }
        //
        //    fn open_autoencoder<P: AsRef<Path>>(p: P) -> Option<MultilayerPerceptron> {
        //        use std::fs::File;
        //        File::open(p).map_err(|_| ()).and_then(|mut f|
        //            bincode::serde::deserialize_from(&mut f, bincode::SizeLimit::Infinite).map_err(|_| ())
        //        ).ok()
        //    }
        //
        //    for _ in widget::Button::new()
        //        .label("Net")
        //        .mid_right_of(ids.autoencoder)
        //        .align_middle_y_of(ids.image_path_btn_autoencoder)
        //        .set(ids.net_path_btn_autoencoder, ui)
        //        {
        //            if let Ok(response) = nfd::open_file_dialog(None, None) {
        //                if let nfd::Response::Okay(path) = response {
        //                    if let x @ Some(..) = open_autoencoder(&path) {
        //                        app.autoencoder_path = path;
        //                        app.net_autoencoder = x;
        //                    }
        //                }
        //            }
        //        }
        //
        //
        //    let mult = 25.0;
        //
        //    //    widget::Image::new()
        //    //        .w_h(7.0 * mult, 10.0 * mult)
        //    //        .mid_left_of(ids.autoencoder)
        //    //        .down(GAP)
        //    //        .set(ids.preview_image_autoencoder, ui);
        //    //
        //    //    widget::Image::new()
        //    //        .w_h(7.0 * mult, 10.0 * mult)
        //    //        .mid_right_of(ids.autoencoder)
        //    //        .down(GAP)
        //    //        .set(ids.preview_image_autoencoder_res, ui);
    }
}

fn classifier_tab(ui: &mut conrod::UiCell, ids: &Ids, app: &mut ClassifierState) {
    use conrod::{widget, Labelable, Positionable, Sizeable, Widget};

    widget::Canvas::new()
        .wh_of(ids.classifier_tab)
        .middle_of(ids.classifier_tab)
        .scroll_kids_vertically()
        .set(ids.classifier_canvas, ui);


    widget::Scrollbar::y_axis(ids.classifier_scrollbar).auto_hide(true).set(ids.tabs_scrollbar, ui);
}

fn mnist_preview_tab(ui: &mut conrod::UiCell, ids: &Ids, app: &mut MnistPreviewState) {
    use conrod::{widget, Labelable, Positionable, Sizeable, Widget};

    widget::Canvas::new()
        .wh_of(ids.mnist_preview_tab)
        .middle_of(ids.mnist_preview_tab)
        .scroll_kids_vertically()
        .set(ids.mnist_preview_canvas, ui);


    widget::Scrollbar::y_axis(ids.mnist_preview_scrollbar).auto_hide(true).set(ids.tabs_scrollbar, ui);
}

fn autoencoder_tab(ui: &mut conrod::UiCell, ids: &Ids, app: &mut AutoencoderState) {
    use conrod::{widget, Labelable, Positionable, Sizeable, Widget};

    widget::Canvas::new()
        .wh_of(ids.autoencoder_tab)
        .middle_of(ids.autoencoder_tab)
        .scroll_kids_vertically()
        .set(ids.autoencoder_canvas, ui);


    widget::Scrollbar::y_axis(ids.autoencoder_scrollbar).auto_hide(true).set(ids.tabs_scrollbar, ui);
}
