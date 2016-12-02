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

pub const WIN_W: u32 = 800;
pub const WIN_H: u32 = 720;


pub struct MulpercWindow {
    pub image_path: String,
    pub image_path_autoencoder: String,
    pub image: Option<Vec<f64>>,
    pub net_path: String,
    pub autoencoder_path: String,
    pub net: Option<(MultilayerPerceptron, HashMap<usize, String>)>,
    pub net_autoencoder: Option<MultilayerPerceptron>,
    pub mnist: Vec<(Vec<f64>, String)>,
    pub mnist_idx: usize
}


impl MulpercWindow {
    pub fn new() -> Self {
        MulpercWindow {
            image_path: "".into(),
            image_path_autoencoder: "".into(),
            image: None,
            net_path: "No net loaded".into(),
            autoencoder_path: "No autoencoder".into(),
            net: None,
            net_autoencoder: None,
            mnist: MnistDigits::default_training_set().unwrap(),
            mnist_idx: 0
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


pub fn image_map<T>(ids: &Ids, preview_image: T, mnist_img: T, preview_image_autoencoder: T, preview_image_autoencoder_res: T) -> conrod::image::Map<T> {
    image_map! {
        (ids.preview_image, preview_image), (ids.mnist_img, mnist_img),
        (ids.preview_image_autoencoder, preview_image_autoencoder), (ids.preview_image_autoencoder_res, preview_image_autoencoder_res)
    }
}


widget_ids! {
    pub struct Ids {
        tabs,
        classifier_canvas,
        autoencoder,

        title,
        title_autoencoder,

        image_path_btn,
        image_path_btn_autoencoder,
        net_path_btn,
        net_path_btn_autoencoder,
        preview_image,
        preview_image_autoencoder,
        preview_image_autoencoder_res,

        label,

        mnist_idx,
        mnist_img,
        mnist_label,

        // Scrollbar
        canvas_scrollbar,
        autoencoder_scrollbar
    }
}


pub fn gui(ui: &mut conrod::UiCell, ids: &Ids, app: &mut MulpercWindow) {
    use conrod::{widget, Labelable, Positionable, Sizeable, Widget};

    const MARGIN: conrod::Scalar = 30.0;
    const GAP: conrod::Scalar = 20.0;

    widget::Tabs::new(&[(ids.classifier_canvas, "Classifier"), (ids.autoencoder, "Autoencoder")])
        .pad(MARGIN).scroll_kids_vertically().set(ids.tabs, ui);

    //    widget::Canvas::new().pad(MARGIN).scroll_kids_vertically().set(ids.classifier_canvas, ui);

    widget::Text::new(&app.net_path)
        .align_text_middle()
        .mid_top_of(ids.classifier_canvas)
        .set(ids.title, ui);

    for _ in widget::Button::new()
        .label("Image")
        .top_left_of(ids.classifier_canvas)
        .down(MARGIN)
        .w_h(130.0, 70.0)
        .set(ids.image_path_btn, ui)
        {
            if let Ok(response) = nfd::open_file_dialog(None, None) {
                if let nfd::Response::Okay(path) = response {
                    app.image_path = path;
                }
            }
        }

    fn open_net<P: AsRef<Path>>(p: P) -> Option<(MultilayerPerceptron, HashMap<usize, String>)> {
        use std::fs::File;
        struct IDontCare;
        impl<T: std::error::Error> From<T> for IDontCare {
            fn from(_: T) -> Self {
                IDontCare
            }
        }
        let res: Result<_, IDontCare> = (|| {
            let mut file = File::open(p)?;
            let NetFile(perc, ntl) = bincode::serde::deserialize_from(&mut file, bincode::SizeLimit::Infinite)?;

            Ok((perc, ntl))
        })();
        res.ok()
    }

    for _ in widget::Button::new()
        .label("Net")
        .mid_right_of(ids.classifier_canvas)
        .align_middle_y_of(ids.image_path_btn)
        .set(ids.net_path_btn, ui)
        {
            if let Ok(response) = nfd::open_file_dialog(None, None) {
                if let nfd::Response::Okay(path) = response {
                    if let x @ Some(..) = open_net(&path) {
                        app.net_path = path;
                        app.net = x;
                    }
                }
            }
        }


    let mult = 25.0;

    widget::Image::new()
        .w_h(7.0 * mult, 10.0 * mult)
        .mid_left_of(ids.classifier_canvas)
        .down(GAP)
        .set(ids.preview_image, ui);

    if let Some((ref perc, ref labels)) = app.net {
        if let Some(ref image) = app.image {
            let out = perc.feed_forward(&*image).0;
            let decoded = &labels[&out.iter().enumerate()
                .max_by(|a, b|
                    a.1.partial_cmp(b.1).unwrap()
                ).unwrap().0];

            widget::Text::new(&decoded)
                .align_text_middle()
                .mid_right_with_margin_on(ids.classifier_canvas, MARGIN)
                .align_middle_y_of(ids.preview_image)
                .font_size(150)
                .set(ids.label, ui);
        }
    }

    for val in widget::NumberDialer::new(app.mnist_idx as f64, 0f64, 59999f64, 0)
        .w_h(130.0, 70.0)
        .down(GAP)
        .align_middle_x_of(ids.classifier_canvas)
        .set(ids.mnist_idx, ui)
        {
            app.mnist_idx = val as usize;
        }

    widget::Image::new()
        .w_h(280.0, 280.0)
        .down(GAP)
        .align_middle_x_of(ids.classifier_canvas)
        .set(ids.mnist_img, ui);

    widget::Text::new(&app.mnist[app.mnist_idx].1)
        .down(GAP)
        .align_middle_x_of(ids.classifier_canvas)
        .set(ids.mnist_label, ui);


    ///////// AUTOENCODER //////////

    widget::Text::new(&app.autoencoder_path)
        .align_text_middle()
        .mid_top_of(ids.autoencoder)
        .set(ids.title_autoencoder, ui);

    for _ in widget::Button::new()
        .label("Image")
        .top_left_of(ids.autoencoder)
        .down(MARGIN)
        .w_h(130.0, 70.0)
        .set(ids.image_path_btn_autoencoder, ui)
        {
            if let Ok(response) = nfd::open_file_dialog(None, None) {
                if let nfd::Response::Okay(path) = response {
                    app.image_path_autoencoder = path;
                }
            }
        }

    fn open_autoencoder<P: AsRef<Path>>(p: P) -> Option<MultilayerPerceptron> {
        use std::fs::File;
        File::open(p).map_err(|_|()).and_then(|mut f|
            bincode::serde::deserialize_from(&mut f, bincode::SizeLimit::Infinite).map_err(|_|())
        ).ok()
    }

    for _ in widget::Button::new()
        .label("Net")
        .mid_right_of(ids.autoencoder)
        .align_middle_y_of(ids.image_path_btn_autoencoder)
        .set(ids.net_path_btn_autoencoder, ui)
        {
            if let Ok(response) = nfd::open_file_dialog(None, None) {
                if let nfd::Response::Okay(path) = response {
                    if let x @ Some(..) = open_autoencoder(&path) {
                        app.autoencoder_path = path;
                        app.net_autoencoder = x;
                    }
                }
            }
        }


    let mult = 25.0;

    widget::Image::new()
        .w_h(7.0 * mult, 10.0 * mult)
        .mid_left_of(ids.autoencoder)
        .down(GAP)
        .set(ids.preview_image_autoencoder, ui);

    widget::Image::new()
        .w_h(7.0 * mult, 10.0 * mult)
        .mid_right_of(ids.autoencoder)
        .down(GAP)
        .set(ids.preview_image_autoencoder_res, ui);

    widget::Scrollbar::y_axis(ids.classifier_canvas).auto_hide(true).set(ids.canvas_scrollbar, ui);
    widget::Scrollbar::y_axis(ids.autoencoder).auto_hide(true).set(ids.autoencoder_scrollbar, ui);
}
