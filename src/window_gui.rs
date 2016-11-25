extern crate rand;

use conrod;
use std;
use nfd;
use multilayer_perceptron::{MultilayerPerceptron, NetFile};
use std::collections::HashMap;
use std::path::Path;
use bincode;
use na::Iterable;

pub const WIN_W: u32 = 400;
pub const WIN_H: u32 = 600;


pub struct MulpercWindow {
    pub image_path: String,
    pub image: Option<Vec<f64>>,
    pub net_path: String,
    pub net: Option<(MultilayerPerceptron, HashMap<usize, String>)>
}


impl MulpercWindow {
    pub fn new() -> Self {
        MulpercWindow {
            image_path: "".into(),
            image: None,
            net_path: "No net loaded".into(),
            net: None,
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


pub fn image_map<T>(ids: &Ids, preview_image: T) -> conrod::image::Map<T> {
    image_map! {
        (ids.preview_image, preview_image)
    }
}


widget_ids! {
    pub struct Ids {
        canvas,

        title,

        image_path_btn,
        net_path_btn,
        preview_image,

        label,

        // Scrollbar
        canvas_scrollbar,

    }
}


pub fn gui(ui: &mut conrod::UiCell, ids: &Ids, app: &mut MulpercWindow) {
    use conrod::{widget, Colorable, Labelable, Positionable, Sizeable, Widget};
    use std::iter::once;

    const MARGIN: conrod::Scalar = 30.0;
    const GAP: conrod::Scalar = 20.0;
    const TITLE_SIZE: conrod::FontSize = 42;
    const SUBTITLE_SIZE: conrod::FontSize = 32;

    widget::Canvas::new().pad(MARGIN).scroll_kids_vertically().set(ids.canvas, ui);

    widget::Text::new(&app.net_path)
        .align_text_middle()
        .mid_top_of(ids.canvas)
        .set(ids.title, ui);

    for press in widget::Button::new()
        .label("Image")
        .top_left_of(ids.canvas)
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
        impl <T: std::error::Error> From<T> for IDontCare {
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

    for press in widget::Button::new()
        .label("Net")
        .mid_right_of(ids.canvas)
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
        .mid_left_of(ids.canvas)
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
                .mid_right_of(ids.canvas)
                .align_middle_y_of(ids.preview_image)
                .font_size(100)
                .set(ids.label, ui);
        }
    }

    widget::Scrollbar::y_axis(ids.canvas).auto_hide(true).set(ids.canvas_scrollbar, ui);
}
